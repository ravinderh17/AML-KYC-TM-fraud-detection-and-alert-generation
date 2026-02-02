# Databricks notebook source
# MAGIC %sql
# MAGIC SHOW CATALOGS;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE CATALOG IF NOT EXISTS aml_catalog1;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS aml_catalog1.bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS aml_catalog1.silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS aml_catalog1.gold;
# MAGIC
# MAGIC -- Volume where you will upload raw files
# MAGIC CREATE VOLUME IF NOT EXISTS aml_catalog1.bronze.raw;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW CATALOGS;

# COMMAND ----------

# MAGIC     %sql
# MAGIC     SHOW VOLUMES IN aml_catalog1.bronze;

# COMMAND ----------

# MAGIC %sql
# MAGIC LIST '/Volumes/aml_catalog1/bronze/raw';

# COMMAND ----------


from pyspark.sql import functions as F

CATALOG = "aml_catalog1"      # change to "main" if using fallback
BRONZE_SCHEMA = "bronze"     # change to "aml_bronze" if fallback
VOLUME = "raw"

raw_path = f"/Volumes/aml_catalog1/bronze/raw/aml_transactions_synthetic_gold_10500_v1.csv"

df_raw = (
    spark.read
      .option("header", "true")
      .option("multiLine", "true")    # JSON blobs can have quotes/commas [1](https://capgemini-my.sharepoint.com/personal/somya_sharma_capgemini_com1/_layouts/15/Doc.aspx?sourcedoc=%7BA3114745-0107-4DAD-AF21-853A8F9B5EB7%7D&file=aml_transactions_raw%201.csv&action=default&mobileredirect=true)
      .option("quote", "\"")
      .option("escape", "\"")
      .option("mode", "PERMISSIVE")
      .csv(raw_path)
)

df_raw.printSchema()
display(df_raw)


# COMMAND ----------


from pyspark.sql import functions as F

CATALOG = "aml_catalog1"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

RAW_VOLUME_PATH = f"/Volumes/{CATALOG}/{BRONZE_SCHEMA}/raw"
RAW_FILE = "aml_transactions_synthetic_gold_10500_v1.csv"

raw_path = f"{RAW_VOLUME_PATH}/{RAW_FILE}"

BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.transactions_bronze"
SILVER_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.transactions_silver"
ALERTS_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.tx_alerts"

print("Raw file path:", raw_path)
print("Bronze table:", BRONZE_TABLE)
print("Silver table:", SILVER_TABLE)
print("Alerts table:", ALERTS_TABLE)

# COMMAND ----------


df_raw = (
    spark.read
      .option("header", "true")
      .option("multiLine", "true")
      .option("quote", "\"")
      .option("escape", "\"")
      .option("mode", "PERMISSIVE")
      .csv(raw_path)
)

df_raw.printSchema()
display(df_raw.limit(10))


# COMMAND ----------


df_bronze = (
    df_raw
    .withColumn("_ingest_ts", F.current_timestamp())
    .withColumn("_source_file", F.col("_metadata.file_path"))  # ✅ UC supported
)

(df_bronze.write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(BRONZE_TABLE)
)

print("Saved Bronze:", BRONZE_TABLE)
display(spark.table(BRONZE_TABLE))


# COMMAND ----------


bronze = spark.table(BRONZE_TABLE)

print("Bronze row count:", bronze.count())
bronze.printSchema()

display(
    bronze.select(
        "transaction_id", "account_id", "amount", "currency_code",
        "booking_ts", "_ingest_ts", "_source_file"
    ))

core_cols = ["transaction_id","account_id","amount","currency_code","direction","transaction_type","booking_ts"]
nulls = bronze.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in core_cols])
display(nulls)

# COMMAND ----------


tx = spark.table(BRONZE_TABLE)

UNKNOWN_TOKENS = ["UNKNOWN", "NULL", "null", "N/A", "na", ""]

string_cols = [c for (c,t) in tx.dtypes if t == "string"]

tx1 = tx
for c in string_cols:
    tx1 = tx1.withColumn(
        c,
        F.when(F.col(c).isNull() | F.trim(F.col(c)).isin(UNKNOWN_TOKENS), F.lit(None))
         .otherwise(F.col(c))
    )

display(tx1.select("counterparty_party_id","originator_party_id","reference"))

# COMMAND ----------


TEXT_COLS = ["reference","pm_raw_payload","pm_parsed_payload_json",
             "iso_parties_by_role_json","screening_events_json","geo_events_json"]

tx2 = tx1
for c in [x for x in TEXT_COLS if x in tx2.columns]:
    tx2 = tx2.withColumn(c, F.regexp_replace(F.col(c), r"[\r\n\t]", " "))

display(tx2.select("reference","screening_events_json").limit(5))


# COMMAND ----------


CODE_COLS = ["currency_code","direction","channel","status","transaction_type",
             "source_system","acct_currency_code","pm_direction","pm_message_standard"]

tx3 = tx2
for c in [x for x in CODE_COLS if x in tx3.columns]:
    tx3 = tx3.withColumn(c, F.upper(F.trim(F.col(c))))

display(tx3.select("currency_code","direction","channel","transaction_type","status").limit(30))


# COMMAND ----------


BOOL_COLS = [c for c in tx3.columns if c.endswith("_flag")] + [
    "is_deleted","acct_is_deleted","pm_is_deleted",
    "orig_is_deleted_x","orig_is_deleted_y",
    "ben_is_deleted_x","ben_is_deleted_y","cp_is_deleted"
]

tx4 = tx3
for c in [x for x in BOOL_COLS if x in tx4.columns]:
    tx4 = tx4.withColumn(
        c,
        F.when(F.upper(F.col(c)).isin("TRUE"), F.lit(True))
         .when(F.upper(F.col(c)).isin("FALSE"), F.lit(False))
         .otherwise(F.lit(None))
    )

display(tx4.select("is_deleted","orig_pep_flag","orig_sanctions_flag","ben_pep_flag","ben_sanctions_flag").limit(30))


# COMMAND ----------


tx5 = (
    tx4
    .withColumn("amount_clean_str", F.regexp_replace(F.col("amount"), r"[^0-9\.\-]", ""))
    .withColumn("amount_num", F.col("amount_clean_str").cast("decimal(18,2)"))
    .withColumn("dq_amount_invalid", (F.col("amount_num").isNull() | (F.col("amount_num") < 0)).cast("int"))
)

display(tx5.select("amount","amount_clean_str","amount_num","dq_amount_invalid").limit(30))


# COMMAND ----------


from pyspark.sql import functions as F

TS_COLS = [
    "booking_ts","created_ts","updated_ts","record_effective_start_ts",
    "acct_created_ts","acct_updated_ts",
    "orig_created_ts_x","orig_updated_ts_x","orig_created_ts_y","orig_updated_ts_y",
    "ben_created_ts_x","ben_updated_ts_x","ben_created_ts_y","ben_updated_ts_y",
    "pm_created_ts","pm_updated_ts",
    "orig_profile_ts","ben_profile_ts","pm_record_effective_start_ts"
]

def add_parsed_ts(df, colname: str):
    norm_col = f"__norm_{colname}"
    jul_key_col = f"__julkey_{colname}"
    jul_ts_col  = f"__jults_{colname}"

    # ----------------------------
    # 0) Normalize + NULL-out blanks
    # ----------------------------
    df = df.withColumn(norm_col, F.trim(F.col(colname)))

    # convert empty string -> null (very important in ANSI mode)
    df = df.withColumn(norm_col, F.when(F.length(F.col(norm_col)) == 0, F.lit(None)).otherwise(F.col(norm_col)))

    # ----------------------------
    # 1) Standard normalizations
    # ----------------------------
    df = df.withColumn(norm_col, F.regexp_replace(F.col(norm_col), r"(UTC|GMT)", ""))
    df = df.withColumn(norm_col, F.regexp_replace(F.col(norm_col), "_", " "))
    df = df.withColumn(norm_col, F.regexp_replace(F.col(norm_col), r"[Tt]", " "))  # remove ISO 'T'
    df = df.withColumn(norm_col, F.regexp_replace(F.col(norm_col), r"([Zz])$", ""))  # drop trailing Z
    df = df.withColumn(norm_col, F.trim(F.col(norm_col)))
    df = df.withColumn(norm_col, F.when(F.length(F.col(norm_col)) == 0, F.lit(None)).otherwise(F.col(norm_col)))

    # ----------------------------
    # 2) Scientific notation fix (2.02601E+13)
    # ----------------------------
    df = df.withColumn(
        norm_col,
        F.when(
            F.col(norm_col).rlike(r"^[0-9]+\.[0-9]+[eE]\+[0-9]+$"),
            F.format_string("%.0f", F.col(norm_col).cast("double"))
        ).otherwise(F.col(norm_col))
    )

    # ----------------------------
    # 3) time-only like 45:00.0 -> force NULL timestamp
    # ----------------------------
    time_only = F.col(norm_col).rlike(r"^\d{1,2}:\d{2}(\.\d+)?$")

    # ----------------------------
    # 4) Julian-like yyyy-DDD HH:mm:ss (e.g., 2026-007 15:00:00)
    #    IMPORTANT: use try_to_date, not to_date
    # ----------------------------
    # Extract pieces (empty if no match)
    jul_year = F.regexp_extract(F.col(norm_col), r"^(\d{4})-(\d{3})", 1)
    jul_doy  = F.regexp_extract(F.col(norm_col), r"^(\d{4})-(\d{3})", 2)
    jul_time = F.regexp_extract(F.col(norm_col), r"^\d{4}-\d{3}\s+(\d{2}:\d{2}:\d{2})", 1)

    # Build yyyyDDD only if regex matched
    df = df.withColumn(
        jul_key_col,
        F.when((jul_year != "") & (jul_doy != ""), F.concat(jul_year, jul_doy)).otherwise(F.lit(None))
    )

    # Convert yyyyDDD -> date using try_to_date (safe)
    df = df.withColumn(
        "__juldate_tmp",
        F.expr(f"try_to_date(`{jul_key_col}`, 'yyyyDDD')")
    )

    # Build julian timestamp string only if date and time exist
    df = df.withColumn(
        jul_ts_col,
        F.when(
            (F.col("__juldate_tmp").isNotNull()) & (jul_time != ""),
            F.concat_ws(" ", F.col("__juldate_tmp").cast("string"), F.lit(jul_time))
        ).otherwise(F.lit(None))
    )

    # ----------------------------
    # 5) Parse timestamps safely
    #    - First try raw (handles +05:30 / +00:00)
    #    - Then try normalized
    #    - Then explicit formats (NO 'T' patterns!)
    # ----------------------------
    parsed = F.coalesce(
        F.expr(f"try_to_timestamp(`{colname}`)"),
        F.expr(f"try_to_timestamp(`{norm_col}`)"),

        F.expr(f"try_to_timestamp(`{norm_col}`, 'dd-MM-yyyy HH:mm')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'dd-MM-yyyy HH:mm:ss')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'dd-MM-yyyy')"),

        F.expr(f"try_to_timestamp(`{norm_col}`, 'MM/dd/yyyy HH:mm:ss')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'MM/dd/yyyy hh:mm:ss a')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'MM/dd/yyyy')"),

        F.expr(f"try_to_timestamp(`{norm_col}`, 'dd/MM/yyyy HH:mm:ss')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'dd/MM/yyyy')"),

        F.expr(f"try_to_timestamp(`{norm_col}`, 'MMM dd yyyy HH:mm:ss')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'MMM-dd-yyyy HH:mm:ss')"),

        F.expr(f"try_to_timestamp(`{norm_col}`, 'yyyy.MM.dd HH:mm:ss')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'yyyy/MM/dd HH:mm:ss')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'yyyy/MM/dd HH:mm')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'yyyy-MM-dd HH:mm:ss')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'yyyy-MM-dd HH:mm')"),

        F.expr(f"try_to_timestamp(`{norm_col}`, 'yyyyMMdd')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'yyyyMMdd HHmmss')"),
        F.expr(f"try_to_timestamp(`{norm_col}`, 'yyyyMMdd HHmm')"),

        # Julian fallback
        F.expr(f"try_to_timestamp(`{jul_ts_col}`, 'yyyy-MM-dd HH:mm:ss')")
    )

    df = df.withColumn(f"{colname}_ts", F.when(time_only, F.lit(None)).otherwise(parsed))

    # cleanup helper columns (keep __norm_* for debugging)
    df = df.drop(jul_key_col, jul_ts_col, "__juldate_tmp")
    return df


tx6 = tx5
for c in [x for x in TS_COLS if x in tx6.columns]:
    tx6 = add_parsed_ts(tx6, c)

tx6 = tx6.withColumn("dq_booking_ts_unparsed", F.col("booking_ts_ts").isNull().cast("int"))

display(tx6.select("booking_ts", "__norm_booking_ts", "booking_ts_ts", "dq_booking_ts_unparsed").limit(60))


# COMMAND ----------


display(
    tx6.select(
        "transaction_id",
        "booking_ts",
        "__norm_booking_ts",
        "booking_ts_ts",
        "dq_booking_ts_unparsed"
    )
    .orderBy(F.desc("dq_booking_ts_unparsed"))
    .limit(50)
)


# COMMAND ----------

 
# Real Time Currency Conversion
 
import requests
from pyspark.sql import functions as F
 
# 1) Fetch daily rates with base USD
er_api_url = "https://open.er-api.com/v6/latest/USD"
resp = requests.get(er_api_url, timeout=20)
resp.raise_for_status()
payload = resp.json()  # keys: result, base_code, time_last_update_utc, rates{...}
 
if payload.get("result") != "success":
    raise RuntimeError(f"FX API returned non-success: {payload.get('result')}")
 
rates = payload.get("rates", {})   # e.g., {"AED": 3.6725, "EUR": 0.86, "GBP": 0.75, ...}
 
# 2) Convert to USD-per-currency for math:
#    If 1 USD = R CURRENCY, then 1 CURRENCY = (1/R) USD
fx_list = []
for code, r in rates.items():
    try:
        r = float(r)
        if r > 0:
            fx_list.append((code.strip().upper(), 1.0 / r))  # USD per 1 currency
    except Exception:
        pass
 
# Ensure USD itself is present
fx_list.append(("USD", 1.0))
 
# 3) Spark DataFrame of FX
fx_df = spark.createDataFrame(fx_list, ["currency_code", "usd_rate"])
 
# 4) Normalize currency_code in your transaction DF (safety)
tx6 = tx6.withColumn("currency_code", F.upper(F.trim(F.col("currency_code"))))
 
# Limit FX to only currencies present in your data to avoid skew
needed_ccys_df = tx6.select("currency_code").distinct()
fx_df = fx_df.join(needed_ccys_df, "currency_code", "inner").dropDuplicates(["currency_code"])
 
# 5) Join and compute USD amount
tx7 = (
    tx6.join(fx_df, "currency_code", "left")
       .withColumn("usd_rate", F.coalesce(F.col("usd_rate"), F.lit(1.0)))  # fallback
       .withColumn("amount_usd",
           (F.col("amount_num") * F.col("usd_rate")).cast("decimal(18,2)")
       )
)
 
display(tx7.select("currency_code", "amount_num", "usd_rate", "amount_usd").limit(30))
 
 

# COMMAND ----------


(tx7.write.format("delta")
    .mode("overwrite")
    .saveAsTable(SILVER_TABLE))

print("Saved Silver:", SILVER_TABLE)
display(spark.table(SILVER_TABLE))


# COMMAND ----------

# GOLD | Parameters & load Silver

from pyspark.sql import functions as F
from pyspark.sql.window import Window as W

# Reuse the same names you already defined earlier
CATALOG       = "aml_catalog1"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA   = "gold"

SILVER_TABLE  = f"{CATALOG}.{SILVER_SCHEMA}.transactions_silver"
GOLD_TABLE    = f"{CATALOG}.{GOLD_SCHEMA}.transactions_gold"
ALERTS_TABLE  = f"{CATALOG}.{GOLD_SCHEMA}.tx_alerts_gold"

df = spark.table(SILVER_TABLE)

# Helper columns
df = (
  df
  .withColumn("booking_ts_ts", F.col("booking_ts_ts"))  # already parsed earlier
  .withColumn("booking_date", F.to_date("booking_ts_ts"))
  .withColumn("amount_usd", F.col("amount_usd").cast("decimal(18,2)"))
  .withColumn("is_cash_credit",
              (F.col("direction")=="CREDIT") & (F.col("transaction_type")=="CASH") &
              F.col("booking_ts_ts").isNotNull())
  .withColumn("is_wire_debit",
              (F.col("direction")=="DEBIT") & F.col("transaction_type").isin("WIRE","ACH","SEPA") &
              F.col("booking_ts_ts").isNotNull())
  .withColumn("is_cross_border",
              (F.col("orig_home_jurisdiction_id").isNotNull())
              & (F.col("ben_home_jurisdiction_id").isNotNull())
              & (F.col("orig_home_jurisdiction_id") != F.col("ben_home_jurisdiction_id")))
  .withColumn("screening_text", F.lower(F.coalesce(F.col("screening_events_json"), F.lit(""))))
  .withColumn("geo_text", F.lower(F.coalesce(F.col("geo_events_json"), F.lit(""))))
)


# COMMAND ----------

# GOLD | Daily cash near-threshold features (structuring/smurfing support)

# Tuning knobs (adjust to local policy)
CTR_THRESHOLD_USD  = F.lit(10000.00)   # cash reporting threshold anchor
NEAR_BAND_LOW_USD  = F.lit(8000.00)    # band lower bound (example: 8k)
NEAR_BAND_HIGH_USD = F.lit(10000.00)   # band upper bound (~10k)

df_cash = (
  df
  .withColumn("is_near_threshold",
              F.col("is_cash_credit") &
              (F.col("amount_usd") >= NEAR_BAND_LOW_USD) & (F.col("amount_usd") < NEAR_BAND_HIGH_USD))
  .filter(F.col("is_near_threshold"))
)

daily_agg = (
  df_cash.groupBy("account_id","booking_date")
    .agg(
      F.count("*").alias("near_thr_cash_cnt"),
      F.sum("amount_usd").alias("near_thr_cash_sum_usd"),
      F.countDistinct("originator_party_id").alias("near_thr_cash_dist_depositors")
    )
)

df = (
  df
  .join(daily_agg, ["account_id","booking_date"], "left")
  .fillna({"near_thr_cash_cnt":0,"near_thr_cash_sum_usd":0.0,"near_thr_cash_dist_depositors":0})
)


# COMMAND ----------

# GOLD | 24h prior credit sum & rapid movement features (layering support)

# Window over 24 hours (in seconds) based on ordering by event time
df = df.withColumn("ts_sec", F.col("booking_ts_ts").cast("timestamp").cast("long"))

# Previous 24h credits on the same account
w_24h = (
  W.partitionBy("account_id")
   .orderBy(F.col("ts_sec"))
   .rangeBetween(-24*3600, -1)   # strictly before current txn
)

df = df.withColumn(
    "prev24h_credit_sum_usd",
    F.sum(F.when(F.col("direction")=="CREDIT", F.col("amount_usd")).otherwise(F.lit(0.0))).over(w_24h)
).fillna({"prev24h_credit_sum_usd": 0.0})

# Rapid sequence: number of tx within ±10 minutes window
w_10m = (
  W.partitionBy("account_id")
   .orderBy(F.col("ts_sec"))
   .rangeBetween(-10*60, 10*60)
)

df = df.withColumn("rapid_seq_txn_cnt", F.count("*").over(w_10m))


# COMMAND ----------



# COMMAND ----------

# GOLD | AML Flags

high_risk_geos = F.array(F.lit("RU"), F.lit("IR"), F.lit("SY"))  # example list; extend as needed

# Basic signal helpers
is_high_risk_geo_touch = (
    F.coalesce(F.col("orig_home_jurisdiction_id"), F.lit("")) .isin(*("RU","IR","SY")) |
    F.coalesce(F.col("ben_home_jurisdiction_id"), F.lit(""))  .isin(*("RU","IR","SY")) |
    F.col("geo_text").contains("high_risk_geo")
)

is_round_dollar = ( (F.col("amount_usd") % 100 == 0) | (F.col("amount_usd") % 1000 == 0) )

# --- Structuring / Smurfing ---
flag_structuring = (
  (F.col("is_cash_credit")) &
  (F.col("amount_usd") >= NEAR_BAND_LOW_USD) & (F.col("amount_usd") < NEAR_BAND_HIGH_USD) &
  (F.col("near_thr_cash_cnt") >= 2)
)

flag_smurfing = (
  (F.col("is_cash_credit")) &
  (F.col("amount_usd") >= NEAR_BAND_LOW_USD) & (F.col("amount_usd") < NEAR_BAND_HIGH_USD) &
  (F.col("near_thr_cash_dist_depositors") >= 3)
)

# --- Placement ---
# account age in days from acct_created_ts_ts (parsed earlier in Silver)
acct_age_days = F.datediff(F.col("booking_ts_ts"), F.col("acct_created_ts_ts"))
flag_placement = (
  F.col("is_cash_credit") & (acct_age_days <= 30) & (F.col("amount_usd") >= F.lit(5000.00))
)

# --- Layering ---
flag_layering_pass_through = (
  F.col("is_wire_debit") &
  (F.col("prev24h_credit_sum_usd") >= F.lit(10000.00))
)
flag_layering_round = is_round_dollar & (F.col("transaction_type").isin("WIRE","ACH","SEPA"))

# --- Sanctions / Screening ---
flag_sanctions_hit = (
  (F.coalesce(F.col("orig_sanctions_flag"), F.lit(False)) == True) |
  (F.coalesce(F.col("ben_sanctions_flag"),  F.lit(False)) == True) |
  F.col("screening_text").contains('"decision":"reject"')
)

flag_screening_escalated = F.col("screening_text").contains('"decision":"escalate"')

# --- PEP ---
flag_pep = (
  (F.coalesce(F.col("orig_pep_flag"), F.lit(False)) == True) |
  (F.coalesce(F.col("ben_pep_flag"),  F.lit(False)) == True) |
  F.col("screening_text").contains("pep")
)

# --- Payments (cross-border high value via SWIFT/ISO) ---
flag_high_value_cross_border = (
  F.col("is_cross_border") &
  F.col("pm_message_standard").isin("SWIFT_MT","ISO20022_MX") &
  (F.col("amount_usd") >= F.lit(100000.00))
)

# --- KYC & Account Behavior ---
flag_kyc_gap = (
  (F.col("orig_kyc_level")=="SIMPLIFIED") & (F.col("amount_usd") >= F.lit(10000.00))
) | (
  (F.col("orig_kyc_level")=="ENHANCED") & F.col("is_wire_debit") & is_high_risk_geo_touch & (F.col("amount_usd") >= F.lit(50000.00))
)

flag_dormant_account_activity = (
  (F.col("acct_status")=="DORMANT") & (F.col("status")=="POSTED")
)

flag_high_risk_geo = is_high_risk_geo_touch

# Assemble flags & simple risk score
df_gold = (
  df
  .withColumn("flag_structuring",              flag_structuring.cast("boolean"))
  .withColumn("flag_smurfing",                 flag_smurfing.cast("boolean"))
  .withColumn("flag_placement",                flag_placement.cast("boolean"))
  .withColumn("flag_layering_pass_through",    flag_layering_pass_through.cast("boolean"))
  .withColumn("flag_layering_round",           flag_layering_round.cast("boolean"))
  .withColumn("flag_sanctions_hit",            flag_sanctions_hit.cast("boolean"))
  .withColumn("flag_screening_escalated",      flag_screening_escalated.cast("boolean"))
  .withColumn("flag_pep",                      flag_pep.cast("boolean"))
  .withColumn("flag_high_value_cross_border",  flag_high_value_cross_border.cast("boolean"))
  .withColumn("flag_kyc_gap",                  flag_kyc_gap.cast("boolean"))
  .withColumn("flag_dormant_account_activity", flag_dormant_account_activity.cast("boolean"))
  .withColumn("flag_high_risk_geo",            flag_high_risk_geo.cast("boolean"))
  .withColumn("flag_round_dollar",             is_round_dollar.cast("boolean"))
  .withColumn(
      "risk_score_simple",
      F.coalesce(F.col("flag_sanctions_hit").cast("int")*5, F.lit(0)) +
      F.coalesce(F.col("flag_pep").cast("int")*4, F.lit(0)) +
      F.coalesce(F.col("flag_high_value_cross_border").cast("int")*3, F.lit(0)) +
      F.coalesce(F.col("flag_layering_pass_through").cast("int")*3, F.lit(0)) +
      F.coalesce(F.col("flag_structuring").cast("int")*3, F.lit(0)) +
      F.coalesce(F.col("flag_smurfing").cast("int")*2, F.lit(0)) +
      F.coalesce(F.col("flag_placement").cast("int")*2, F.lit(0)) +
      F.coalesce(F.col("flag_kyc_gap").cast("int")*2, F.lit(0)) +
      F.coalesce(F.col("flag_dormant_account_activity").cast("int")*1, F.lit(0)) +
      F.coalesce(F.col("flag_high_risk_geo").cast("int")*1, F.lit(0)) +
      F.coalesce(F.col("flag_round_dollar").cast("int")*1, F.lit(0))
  )
)


# COMMAND ----------

# GOLD | Build a long-form Alerts table (version-agnostic via stack)

from pyspark.sql import functions as F

# 1) Select only the columns we need + flags
base_cols = [
  "transaction_id", "account_id", "amount_usd", "booking_ts_ts",
  "near_thr_cash_cnt", "near_thr_cash_sum_usd", "near_thr_cash_dist_depositors",
  "prev24h_credit_sum_usd", "risk_score_simple"
]

flag_cols = [
  ("STRUCTURING",               "flag_structuring"),
  ("SMURFING",                  "flag_smurfing"),
  ("PLACEMENT",                 "flag_placement"),
  ("LAYERING_PASS_THROUGH",     "flag_layering_pass_through"),
  ("LAYERING_ROUND",            "flag_layering_round"),
  ("SANCTIONS_HIT",             "flag_sanctions_hit"),
  ("SCREENING_ESCALATED",       "flag_screening_escalated"),
  ("PEP",                       "flag_pep"),
  ("HIGH_VALUE_XBORDER",        "flag_high_value_cross_border"),
  ("KYC_GAP",                   "flag_kyc_gap"),
  ("DORMANT_ACCOUNT_ACTIVITY",  "flag_dormant_account_activity"),
  ("HIGH_RISK_GEO",             "flag_high_risk_geo"),
  ("ROUND_DOLLAR",              "flag_round_dollar"),
]

# 2) Build an expression with stack: each (alert_type, fired_flag) pair becomes a row
# stack(N, 'CODE1', flag1, 'CODE2', flag2, ... )
stack_expr_parts = []
for code, col_name in flag_cols:
    stack_expr_parts.append(f"'{code}'")
    stack_expr_parts.append(f"CAST({col_name} AS INT)")
stack_expr = f"stack({len(flag_cols)}, {', '.join(stack_expr_parts)}) AS (alert_type, fired_flag)"

# 3) Create long format, filter fired_flag = 1
long_df = (
  df_gold
    .select(*(base_cols + [c for _, c in flag_cols]))
    .selectExpr(*base_cols, stack_expr)
    .filter("fired_flag = 1")
    .drop("fired_flag")
)

# 4) Build alert_reason with CASE logic (no lambdas, portable)
alerts_df = (
  long_df
    .withColumn(
      "alert_reason",
      F.when(F.col("alert_type")=="STRUCTURING",
             F.concat_ws(" | ",
                F.lit("Multiple cash deposits near threshold on same day"),
                F.concat(F.lit("count="), F.col("near_thr_cash_cnt")),
                F.concat(F.lit("sum_usd="), F.col("near_thr_cash_sum_usd"))
             ))
       .when(F.col("alert_type")=="SMURFING",
             F.concat_ws(" | ",
                F.lit("Many distinct depositors near threshold on same day"),
                F.concat(F.lit("distinct_depositors="), F.col("near_thr_cash_dist_depositors"))
             ))
       .when(F.col("alert_type")=="PLACEMENT", F.lit("Cash into young account (<=30 days)"))
       .when(F.col("alert_type")=="LAYERING_PASS_THROUGH",
             F.concat_ws(" | ",
                F.lit("Wire debit after large credits within 24h"),
                F.concat(F.lit("prev24h_credit_sum_usd="), F.col("prev24h_credit_sum_usd"))
             ))
       .when(F.col("alert_type")=="LAYERING_ROUND", F.lit("Round-dollar amount on WIRE/ACH/SEPA"))
       .when(F.col("alert_type")=="SANCTIONS_HIT", F.lit("Sanctions flag or screening reject"))
       .when(F.col("alert_type")=="SCREENING_ESCALATED", F.lit("Screening escalation"))
       .when(F.col("alert_type")=="PEP", F.lit("PEP involvement (direct or screening)"))
       .when(F.col("alert_type")=="HIGH_VALUE_XBORDER", F.lit("High-value cross-border via SWIFT/ISO"))
       .when(F.col("alert_type")=="KYC_GAP", F.lit("KYC level vs amount/geo mismatch"))
       .when(F.col("alert_type")=="DORMANT_ACCOUNT_ACTIVITY", F.lit("Posted activity on dormant account"))
       .when(F.col("alert_type")=="HIGH_RISK_GEO", F.lit("High-risk geography (direct/derived)"))
       .when(F.col("alert_type")=="ROUND_DOLLAR", F.lit("Round-dollar anomaly"))
       .otherwise(F.lit(None))
    )
    .select(
      F.col("transaction_id"),
      F.col("account_id"),
      F.col("amount_usd"),
      F.col("booking_ts_ts").alias("booking_ts"),
      F.col("alert_type"),
      F.col("alert_reason"),
      F.col("risk_score_simple").alias("tx_risk_score")
    )
)

# 5) Persist
(
  alerts_df
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(ALERTS_TABLE)
)

print("Saved Alerts:", ALERTS_TABLE, "| rows:", alerts_df.count())
display(spark.table(ALERTS_TABLE).groupBy("alert_type").count().orderBy(F.desc("count")))


# COMMAND ----------



# COMMAND ----------

# GOLD | Persist the Gold table

(
  df_gold
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(GOLD_TABLE)
)

print("Saved Gold:", GOLD_TABLE)
display(spark.table(GOLD_TABLE).select(
  "transaction_id","account_id","amount_usd","booking_ts_ts","risk_score_simple",
  "flag_structuring","flag_smurfing","flag_placement","flag_layering_pass_through",
  "flag_sanctions_hit","flag_pep","flag_high_value_cross_border","flag_kyc_gap",
  "flag_dormant_account_activity","flag_high_risk_geo","flag_round_dollar"
).orderBy(F.desc("risk_score_simple")).limit(100))


# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- COMMAND ----------
# MAGIC -- GOLD | Handy queries (optional)
# MAGIC
# MAGIC -- Highest risk transactions
# MAGIC SELECT *
# MAGIC FROM aml_catalog1.gold.transactions_gold
# MAGIC ORDER BY risk_score_simple DESC
# MAGIC LIMIT 100;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Top alert types by count
# MAGIC SELECT alert_type, COUNT(*) AS cnt
# MAGIC FROM aml_catalog1.gold.tx_alerts_gold
# MAGIC GROUP BY alert_type
# MAGIC ORDER BY cnt DESC;
# MAGIC

# COMMAND ----------


# Inspect a few flag counts
flag_names = [
  "flag_structuring","flag_smurfing","flag_placement",
  "flag_layering_pass_through","flag_layering_round",
  "flag_sanctions_hit","flag_screening_escalated","flag_pep",
  "flag_high_value_cross_border","flag_kyc_gap",
  "flag_dormant_account_activity","flag_high_risk_geo","flag_round_dollar"
]

display(
  df_gold.select([F.sum(F.col(f).cast("int")).alias(f) for f in flag_names])
)


# COMMAND ----------


display(
  df_gold
    .filter((F.col("account_id")=="A10000") & (F.to_date("booking_ts_ts")==F.to_date(F.lit("2026-01-05"))))
    .select("transaction_id","account_id","amount_usd","transaction_type","direction",
            "near_thr_cash_cnt","near_thr_cash_dist_depositors","flag_structuring","flag_smurfing")
    .orderBy("transaction_id")
)


# COMMAND ----------


# Make sure variables in scope are correct
print("GOLD_TABLE:", GOLD_TABLE)
print("ALERTS_TABLE:", ALERTS_TABLE)

# Show tables in the gold schema
spark.sql("SHOW TABLES IN aml_catalog1.gold").show(truncate=False)



# COMMAND ----------

print("df_gold rows:", df_gold.count())

# COMMAND ----------


from pyspark.sql import functions as F

flag_names = [
  "flag_structuring","flag_smurfing","flag_placement",
  "flag_layering_pass_through","flag_layering_round",
  "flag_sanctions_hit","flag_screening_escalated","flag_pep",
  "flag_high_value_cross_border","flag_kyc_gap",
  "flag_dormant_account_activity","flag_high_risk_geo","flag_round_dollar"
]

display(
  df_gold.select([F.sum(F.col(f).cast("int")).alias(f) for f in flag_names])
)
 

# COMMAND ----------

# GOLD | Build a long-form Alerts table (version-agnostic via stack)

from pyspark.sql import functions as F

# 1) Select only the columns we need + flags
base_cols = [
  "transaction_id", "account_id", "amount_usd", "booking_ts_ts",
  "near_thr_cash_cnt", "near_thr_cash_sum_usd", "near_thr_cash_dist_depositors",
  "prev24h_credit_sum_usd", "risk_score_simple"
]

flag_cols = [
  ("STRUCTURING",               "flag_structuring"),
  ("SMURFING",                  "flag_smurfing"),
  ("PLACEMENT",                 "flag_placement"),
  ("LAYERING_PASS_THROUGH",     "flag_layering_pass_through"),
  ("LAYERING_ROUND",            "flag_layering_round"),
  ("SANCTIONS_HIT",             "flag_sanctions_hit"),
  ("SCREENING_ESCALATED",       "flag_screening_escalated"),
  ("PEP",                       "flag_pep"),
  ("HIGH_VALUE_XBORDER",        "flag_high_value_cross_border"),
  ("KYC_GAP",                   "flag_kyc_gap"),
  ("DORMANT_ACCOUNT_ACTIVITY",  "flag_dormant_account_activity"),
  ("HIGH_RISK_GEO",             "flag_high_risk_geo"),
  ("ROUND_DOLLAR",              "flag_round_dollar"),
]

# 2) Build an expression with stack: each (alert_type, fired_flag) pair becomes a row
# stack(N, 'CODE1', flag1, 'CODE2', flag2, ... )
stack_expr_parts = []
for code, col_name in flag_cols:
    stack_expr_parts.append(f"'{code}'")
    stack_expr_parts.append(f"CAST({col_name} AS INT)")
stack_expr = f"stack({len(flag_cols)}, {', '.join(stack_expr_parts)}) AS (alert_type, fired_flag)"

# 3) Create long format, filter fired_flag = 1
long_df = (
  df_gold
    .select(*(base_cols + [c for _, c in flag_cols]))
    .selectExpr(*base_cols, stack_expr)
    .filter("fired_flag = 1")
    .drop("fired_flag")
)

# 4) Build alert_reason with CASE logic (no lambdas, portable)
alerts_df = (
  long_df
    .withColumn(
      "alert_reason",
      F.when(F.col("alert_type")=="STRUCTURING",
             F.concat_ws(" | ",
                F.lit("Multiple cash deposits near threshold on same day"),
                F.concat(F.lit("count="), F.col("near_thr_cash_cnt")),
                F.concat(F.lit("sum_usd="), F.col("near_thr_cash_sum_usd"))
             ))
       .when(F.col("alert_type")=="SMURFING",
             F.concat_ws(" | ",
                F.lit("Many distinct depositors near threshold on same day"),
                F.concat(F.lit("distinct_depositors="), F.col("near_thr_cash_dist_depositors"))
             ))
       .when(F.col("alert_type")=="PLACEMENT", F.lit("Cash into young account (<=30 days)"))
       .when(F.col("alert_type")=="LAYERING_PASS_THROUGH",
             F.concat_ws(" | ",
                F.lit("Wire debit after large credits within 24h"),
                F.concat(F.lit("prev24h_credit_sum_usd="), F.col("prev24h_credit_sum_usd"))
             ))
       .when(F.col("alert_type")=="LAYERING_ROUND", F.lit("Round-dollar amount on WIRE/ACH/SEPA"))
       .when(F.col("alert_type")=="SANCTIONS_HIT", F.lit("Sanctions flag or screening reject"))
       .when(F.col("alert_type")=="SCREENING_ESCALATED", F.lit("Screening escalation"))
       .when(F.col("alert_type")=="PEP", F.lit("PEP involvement (direct or screening)"))
       .when(F.col("alert_type")=="HIGH_VALUE_XBORDER", F.lit("High-value cross-border via SWIFT/ISO"))
       .when(F.col("alert_type")=="KYC_GAP", F.lit("KYC level vs amount/geo mismatch"))
       .when(F.col("alert_type")=="DORMANT_ACCOUNT_ACTIVITY", F.lit("Posted activity on dormant account"))
       .when(F.col("alert_type")=="HIGH_RISK_GEO", F.lit("High-risk geography (direct/derived)"))
       .when(F.col("alert_type")=="ROUND_DOLLAR", F.lit("Round-dollar anomaly"))
       .otherwise(F.lit(None))
    )
    .select(
      F.col("transaction_id"),
      F.col("account_id"),
      F.col("amount_usd"),
      F.col("booking_ts_ts").alias("booking_ts"),
      F.col("alert_type"),
      F.col("alert_reason"),
      F.col("risk_score_simple").alias("tx_risk_score")
    )
)

# 5) Persist
(
  alerts_df
  .write
  .format("delta")
  .mode("overwrite")
  .saveAsTable(ALERTS_TABLE)
)

print("Saved Alerts:", ALERTS_TABLE, "| rows:", alerts_df.count())
display(spark.table(ALERTS_TABLE).groupBy("alert_type").count().orderBy(F.desc("count")))


# COMMAND ----------


print("df_gold rows:", df_gold.count())
alerts_count = spark.table(ALERTS_TABLE).count()
print("alerts table rows:", alerts_count)

display(
  spark.table(ALERTS_TABLE)
       .groupBy("alert_type")
       .count()
       .orderBy(F.desc("count"))
)

# COMMAND ----------


from pyspark.sql import functions as F

GOLD_TABLE = "aml_catalog1.gold.transactions_gold"
ALERTS_TABLE = "aml_catalog1.gold.tx_alerts_gold"

df_gold = spark.table(GOLD_TABLE)
df_alerts = spark.table(ALERTS_TABLE)

print("Gold rows:", df_gold.count())
print("Alerts rows:", df_alerts.count())


# COMMAND ----------


dq_booking = (
    df_gold
    .groupBy("dq_booking_ts_unparsed")
    .count()
    .withColumn(
        "label",
        F.when(F.col("dq_booking_ts_unparsed") == 1, "Unparsed")
         .otherwise("Parsed")
    )
)

display(dq_booking)


# COMMAND ----------


invalid_amounts = (
    spark.table("aml_catalog1.silver.transactions_silver")
    .groupBy("dq_amount_invalid")
    .count()
    .withColumn(
        "status",
        F.when(F.col("dq_amount_invalid") == 1, "Invalid Amount")
         .otherwise("Valid Amount")
    )
)

display(invalid_amounts)


# COMMAND ----------


tx_type_dist = (
    df_gold
    .groupBy("transaction_type")
    .count()
    .orderBy(F.desc("count"))
)

display(tx_type_dist)

# COMMAND ----------


from pyspark.sql import functions as F

GOLD_TABLE   = "aml_catalog1.gold.transactions_gold"
ALERTS_TABLE = "aml_catalog1.gold.tx_alerts_gold"
SILVER_TABLE = "aml_catalog1.silver.transactions_silver"
BRONZE_TABLE = "aml_catalog1.bronze.transactions_bronze"

df_gold   = spark.table(GOLD_TABLE)
df_alerts = spark.table(ALERTS_TABLE)
df_silver = spark.table(SILVER_TABLE)
df_bronze = spark.table(BRONZE_TABLE)

print("Bronze:", df_bronze.count())
print("Silver:", df_silver.count())
print("Gold:", df_gold.count())
print("Alerts:", df_alerts.count())


# COMMAND ----------


display(
  df_gold
    .select(F.col("amount_usd").cast("double").alias("amount_usd"))
    .where(F.col("amount_usd").isNotNull())
)


# COMMAND ----------


dq_booking = (
    df_silver
    .groupBy("dq_booking_ts_unparsed")
    .count()
    .withColumn("status",
        F.when(F.col("dq_booking_ts_unparsed")==1, F.lit("Unparsed booking_ts"))
         .otherwise(F.lit("Parsed booking_ts"))
    )
    .select("status","count")
)

display(dq_booking)


# COMMAND ----------


dq_amount = (
    df_silver
    .groupBy("dq_amount_invalid")
    .count()
    .withColumn("status",
        F.when(F.col("dq_amount_invalid")==1, F.lit("Invalid amount"))
         .otherwise(F.lit("Valid amount"))
    )
    .select("status","count")
)

display(dq_amount)


# COMMAND ----------


tx_type_dist = (
    df_gold.groupBy("transaction_type")
    .count()
    .orderBy(F.desc("count"))
)

display(tx_type_dist)


# COMMAND ----------


channel_mix = (
    df_gold.groupBy("channel")
    .count()
    .orderBy(F.desc("count"))
)

display(channel_mix)


# COMMAND ----------


daily_trend = (
    df_gold.groupBy("booking_date")
    .agg(
        F.count("*").alias("tx_count"),
        F.sum("amount_usd").alias("total_usd")
    )
    .orderBy("booking_date")
)

display(daily_trend)


# COMMAND ----------


near_thr_daily = (
    df_gold
    .groupBy("booking_date")
    .agg(
        F.sum(F.col("near_thr_cash_cnt")).alias("near_thr_cash_cnt_total"),
        F.sum(F.col("near_thr_cash_sum_usd")).alias("near_thr_cash_sum_usd_total")
    )
    .orderBy("booking_date")
)

display(near_thr_daily)


# COMMAND ----------


risk_dist = (
    df_gold.groupBy("risk_score_simple")
    .count()
    .orderBy("risk_score_simple")
)

display(risk_dist)


# COMMAND ----------


alert_type_dist = (
    df_alerts.groupBy("alert_type")
    .count()
    .orderBy(F.desc("count"))
)

display(alert_type_dist)


# COMMAND ----------


alert_risk_heat = (
    df_alerts
    .groupBy("alert_type", "tx_risk_score")
    .count()
)

display(alert_risk_heat)


# COMMAND ----------


top_accounts = (
    df_gold.groupBy("account_id")
    .agg(
        F.sum("risk_score_simple").alias("total_risk"),
        F.count("*").alias("tx_count"),
        F.sum("amount_usd").alias("total_usd")
    )
    .orderBy(F.desc("total_risk"))
)

display(top_accounts.limit(15))


# COMMAND ----------


# mark if transaction has any alert
tx_has_alert = (
    df_alerts.select("transaction_id")
    .distinct()
    .withColumn("has_alert", F.lit(1))
)

scatter_df = (
    df_gold.select("transaction_id","amount_usd","risk_score_simple","transaction_type")
    .join(tx_has_alert, "transaction_id", "left")
    .fillna({"has_alert": 0})
)

display(scatter_df)


# COMMAND ----------


from pyspark.sql import functions as F

# 1) Base aggregation
alert_risk_counts = (
    df_alerts
    .select(
        F.col("alert_type"),
        F.col("tx_risk_score").cast("int").alias("tx_risk_score")
    )
    .groupBy("alert_type", "tx_risk_score")
    .count()
)

# 2) Build a COMPLETE risk score axis (min..max)
min_max = alert_risk_counts.agg(
    F.min("tx_risk_score").alias("min_rs"),
    F.max("tx_risk_score").alias("max_rs")
).collect()[0]

min_rs = int(min_max["min_rs"])
max_rs = int(min_max["max_rs"])

risk_scores = (
    spark.range(min_rs, max_rs + 1)
    .withColumnRenamed("id", "tx_risk_score")
    .select(F.col("tx_risk_score").cast("int"))
)

# 3) Get full list of alert types
alert_types = df_alerts.select("alert_type").distinct()

# 4) Create FULL GRID (alert_type x tx_risk_score)
full_grid = alert_types.crossJoin(risk_scores)

# 5) Join counts onto grid and fill missing with 0
alert_risk_full = (
    full_grid
    .join(alert_risk_counts, ["alert_type", "tx_risk_score"], "left")
    .fillna({"count": 0})
)

# 6) Make zeros visible with faint color (so no "blank-looking" cells)
#    We keep the real count AND a color column:
alert_risk_full = alert_risk_full.withColumn(
    "count_for_color",
    F.when(F.col("count") == 0, F.lit(0.1)).otherwise(F.col("count").cast("double"))
)

display(alert_risk_full.orderBy("alert_type", "tx_risk_score"))

# COMMAND ----------


import matplotlib.pyplot as plt
import pandas as pd

# Use the "alert_risk_full" created earlier
pdf = (
    alert_risk_full
    .select("alert_type", "tx_risk_score", "count")
    .toPandas()
)

pivot = pdf.pivot(index="alert_type", columns="tx_risk_score", values="count").fillna(0)

plt.figure(figsize=(12, 6))
plt.imshow(pivot.values, aspect="auto", cmap="YlOrRd")  # professional palette

plt.colorbar(label="Alert Count")
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)

# Annotate each cell with actual count
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = int(pivot.values[i, j])
        plt.text(j, i, str(val), ha='center', va='center',
                 color='black' if val < pivot.values.max()*0.6 else 'white', fontsize=9)

plt.xlabel("tx_risk_score")
plt.ylabel("alert_type")
plt.title("Alert Type vs Risk Score Heatmap (Counts)")
plt.tight_layout()
plt.show()