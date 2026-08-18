# ============================================================
# PEARLS AQI PREDICTOR
# Phase 2 — Exploratory Data Analysis
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = Path("data/processed/aqi_features.csv")

OUTPUT_DIR = Path("reports/eda")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
sns.set_theme(style="whitegrid")

# Pandas display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load the processed AQI dataset."""

    df = pd.read_csv(INPUT_PATH)

    df["date"] = pd.to_datetime(df["date"])

    return df


# ============================================================
# BASIC DATA INFORMATION
# ============================================================

def basic_information(df):
    """Print basic information about the dataset."""

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head().to_string())


# ============================================================
# MISSING VALUES
# ============================================================

def missing_values(df):
    """Check missing values."""

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = df.isna().sum()

    missing_percentage = (
        df.isna().mean() * 100
    ).round(2)

    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_percentage": missing_percentage
    })

    summary = summary[
        summary["missing_count"] > 0
    ]

    if summary.empty:
        print("No missing values found.")

    else:
        print(summary.to_string())


# ============================================================
# DUPLICATES
# ============================================================

def check_duplicates(df):
    """Check duplicate rows and dates."""

    print("\n" + "=" * 60)
    print("DUPLICATES")
    print("=" * 60)

    duplicate_rows = df.duplicated().sum()

    duplicate_dates = df["date"].duplicated().sum()

    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Duplicate dates: {duplicate_dates}")


# ============================================================
# DATE INFORMATION
# ============================================================

def date_analysis(df):
    """Analyze date range and gaps."""

    print("\n" + "=" * 60)
    print("DATE ANALYSIS")
    print("=" * 60)

    print("Start date:", df["date"].min())
    print("End date:  ", df["date"].max())

    date_difference = (
        df["date"].sort_values().diff().dt.days
    )

    gaps = date_difference[
        date_difference > 1
    ]

    print(f"\nNumber of date gaps: {len(gaps)}")

    if len(gaps) > 0:

        print("\nGaps found:")

        for index in gaps.index:

            current_date = df.loc[index, "date"]
            previous_date = df.loc[index - 1, "date"]

            missing_days = (
                current_date - previous_date
            ).days - 1

            print(
                f"{previous_date.date()} → "
                f"{current_date.date()} "
                f"({missing_days} missing day(s))"
            )


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_statistics(df):
    """Display numerical summary."""

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    print(
        df[numeric_columns]
        .describe()
        .T
        .to_string()
    )


# ============================================================
# AQI DISTRIBUTION
# ============================================================

def plot_aqi_distribution(df):

    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="AQI",
        bins=30,
        kde=True
    )

    plt.title("Distribution of AQI")
    plt.xlabel("AQI")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "aqi_distribution.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# AQI BOXPLOT
# ============================================================

def plot_aqi_boxplot(df):

    plt.figure(figsize=(10, 4))

    sns.boxplot(
        x=df["AQI"]
    )

    plt.title("AQI Boxplot")
    plt.xlabel("AQI")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "aqi_boxplot.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# AQI OVER TIME
# ============================================================

def plot_aqi_over_time(df):

    plt.figure(figsize=(15, 6))

    plt.plot(
        df["date"],
        df["AQI"],
        linewidth=1
    )

    plt.title("AQI Over Time")
    plt.xlabel("Date")
    plt.ylabel("AQI")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "aqi_over_time.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# 30-DAY AQI ROLLING TREND
# ============================================================

def plot_aqi_rolling_mean(df):

    rolling_aqi = (
        df["AQI"]
        .rolling(window=30)
        .mean()
    )

    plt.figure(figsize=(15, 6))

    plt.plot(
        df["date"],
        df["AQI"],
        alpha=0.3,
        label="Daily AQI"
    )

    plt.plot(
        df["date"],
        rolling_aqi,
        linewidth=2,
        label="30-Day Rolling Mean"
    )

    plt.title(
        "AQI Trend with 30-Day Rolling Mean"
    )

    plt.xlabel("Date")
    plt.ylabel("AQI")

    plt.legend()

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "aqi_rolling_trend.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# POLLUTANT DISTRIBUTIONS
# ============================================================

def plot_pollutant_distributions(df):

    pollutants = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3"
    ]

    for column in pollutants:

        plt.figure(figsize=(9, 4))

        sns.histplot(
            data=df,
            x=column,
            bins=30,
            kde=True
        )

        plt.title(
            f"Distribution of {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()

        filename = (
            column
            .replace(".", "")
            .replace(" ", "_")
            .lower()
        )

        plt.savefig(
            OUTPUT_DIR / f"{filename}_distribution.png",
            dpi=300
        )

        plt.show()

        plt.close()


# ============================================================
# WEATHER DISTRIBUTIONS
# ============================================================

def plot_weather_distributions(df):

    weather_features = [
        "Temperature",
        "Humidity",
        "Precipitation"
    ]

    for column in weather_features:

        plt.figure(figsize=(9, 4))

        sns.histplot(
            data=df,
            x=column,
            bins=30,
            kde=True
        )

        plt.title(
            f"Distribution of {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.tight_layout()

        filename = (
            column
            .replace(".", "")
            .replace(" ", "_")
            .lower()
        )

        plt.savefig(
            OUTPUT_DIR / f"{filename}_distribution.png",
            dpi=300
        )

        plt.show()

        plt.close()


# ============================================================
# POLLUTANTS OVER TIME
# ============================================================

def plot_pollutants_over_time(df):

    pollutants = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3"
    ]

    for column in pollutants:

        plt.figure(figsize=(15, 4))

        plt.plot(
            df["date"],
            df[column],
            linewidth=1
        )

        plt.title(
            f"{column} Over Time"
        )

        plt.xlabel("Date")
        plt.ylabel(column)

        plt.xticks(rotation=45)

        plt.tight_layout()

        filename = (
            column
            .replace(".", "")
            .replace(" ", "_")
            .lower()
        )

        plt.savefig(
            OUTPUT_DIR / f"{filename}_over_time.png",
            dpi=300
        )

        plt.show()

        plt.close()


# ============================================================
# AQI VS POLLUTANTS
# ============================================================

def plot_aqi_vs_pollutants(df):

    pollutants = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3"
    ]

    for column in pollutants:

        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            data=df,
            x=column,
            y="AQI",
            alpha=0.6
        )

        plt.title(
            f"AQI vs {column}"
        )

        plt.xlabel(column)
        plt.ylabel("AQI")

        plt.tight_layout()

        filename = (
            column
            .replace(".", "")
            .replace(" ", "_")
            .lower()
        )

        plt.savefig(
            OUTPUT_DIR / f"aqi_vs_{filename}.png",
            dpi=300
        )

        plt.show()

        plt.close()


# ============================================================
# AQI VS WEATHER
# ============================================================

def plot_aqi_vs_weather(df):

    weather_features = [
        "Temperature",
        "Humidity",
        "Precipitation"
    ]

    for column in weather_features:

        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            data=df,
            x=column,
            y="AQI",
            alpha=0.6
        )

        plt.title(
            f"AQI vs {column}"
        )

        plt.xlabel(column)
        plt.ylabel("AQI")

        plt.tight_layout()

        filename = (
            column
            .replace(".", "")
            .replace(" ", "_")
            .lower()
        )

        plt.savefig(
            OUTPUT_DIR / f"aqi_vs_{filename}.png",
            dpi=300
        )

        plt.show()

        plt.close()


# ============================================================
# CORRELATION WITH AQI
# ============================================================

def aqi_correlations(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    correlations = (
        df[numeric_columns]
        .corr()["AQI"]
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 60)
    print("CORRELATION WITH CURRENT AQI")
    print("=" * 60)

    print(
        correlations.to_string()
    )

    correlations.to_csv(
        OUTPUT_DIR / "aqi_correlations.csv"
    )


# ============================================================
# CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(df):

    features = [
        "AQI",
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "Temperature",
        "Humidity",
        "Precipitation",
        "AQI_lag_1",
        "AQI_lag_2",
        "AQI_lag_3",
        "AQI_lag_7",
        "AQI_change_1d"
    ]

    correlation_matrix = (
        df[features]
        .corr()
    )

    plt.figure(
        figsize=(14, 10)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title(
        "Feature Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# MONTHLY AQI
# ============================================================

def monthly_analysis(df):

    monthly = (
        df.groupby("month")["AQI"]
        .agg(
            [
                "mean",
                "median",
                "min",
                "max"
            ]
        )
    )

    print("\n" + "=" * 60)
    print("AQI BY MONTH")
    print("=" * 60)

    print(
        monthly.to_string()
    )

    monthly.to_csv(
        OUTPUT_DIR / "monthly_aqi.csv"
    )

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=(
            df.groupby(
                "month",
                as_index=False
            )["AQI"]
            .mean()
        ),
        x="month",
        y="AQI"
    )

    plt.title(
        "Average AQI by Month"
    )

    plt.xlabel("Month")
    plt.ylabel("Average AQI")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "aqi_by_month.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# DAY OF WEEK ANALYSIS
# ============================================================

def day_of_week_analysis(df):

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekly = (
        df.groupby("day_of_week")["AQI"]
        .mean()
    )

    weekly.index = [
        day_names[i]
        for i in weekly.index
    ]

    print("\n" + "=" * 60)
    print("AQI BY DAY OF WEEK")
    print("=" * 60)

    print(
        weekly.to_string()
    )

    weekly.to_csv(
        OUTPUT_DIR / "aqi_by_day_of_week.csv"
    )

    plt.figure(figsize=(10, 5))

    weekly.plot(
        kind="bar"
    )

    plt.title(
        "Average AQI by Day of Week"
    )

    plt.xlabel("Day")
    plt.ylabel("Average AQI")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "aqi_by_day_of_week.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# YEARLY AQI
# ============================================================

def yearly_analysis(df):

    yearly = (
        df.groupby("year")["AQI"]
        .agg(
            [
                "mean",
                "median",
                "min",
                "max"
            ]
        )
    )

    print("\n" + "=" * 60)
    print("AQI BY YEAR")
    print("=" * 60)

    print(
        yearly.to_string()
    )

    yearly.to_csv(
        OUTPUT_DIR / "yearly_aqi.csv"
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=(
            df.groupby(
                "year",
                as_index=False
            )["AQI"]
            .mean()
        ),
        x="year",
        y="AQI"
    )

    plt.title(
        "Average AQI by Year"
    )

    plt.xlabel("Year")
    plt.ylabel("Average AQI")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "aqi_by_year.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# AQI LAG RELATIONSHIPS
# ============================================================

def plot_aqi_lags(df):

    lag_features = [
        "AQI_lag_1",
        "AQI_lag_2",
        "AQI_lag_3",
        "AQI_lag_7"
    ]

    for column in lag_features:

        temp = df[
            ["AQI", column]
        ].dropna()

        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            data=temp,
            x=column,
            y="AQI",
            alpha=0.5
        )

        plt.title(
            f"Current AQI vs {column}"
        )

        plt.xlabel(column)
        plt.ylabel("Current AQI")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"aqi_vs_{column}.png",
            dpi=300
        )

        plt.show()

        plt.close()


# ============================================================
# AQI CHANGE
# ============================================================

def plot_aqi_change(df):

    plt.figure(figsize=(10, 5))

    sns.histplot(
        data=df,
        x="AQI_change_1d",
        bins=40,
        kde=True
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.title(
        "Distribution of Daily AQI Change"
    )

    plt.xlabel(
        "AQI Change from Previous Day"
    )

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "aqi_change_distribution.png",
        dpi=300
    )

    plt.show()

    plt.close()


# ============================================================
# TARGET ANALYSIS
# ============================================================

def target_analysis(df):

    targets = [
        "AQI_t+1",
        "AQI_t+2",
        "AQI_t+3"
    ]

    print("\n" + "=" * 60)
    print("FUTURE AQI TARGETS")
    print("=" * 60)

    target_summary = (
        df[
            ["AQI"] + targets
        ]
        .describe()
        .T
    )

    print(
        target_summary.to_string()
    )

    target_correlations = (
        df[
            ["AQI"] + targets
        ]
        .corr()["AQI"]
    )

    print("\nCorrelation with current AQI:")

    print(
        target_correlations.to_string()
    )

    target_correlations.to_csv(
        OUTPUT_DIR / "target_correlations.csv"
    )

    for target in targets:

        temp = df[
            ["AQI", target]
        ].dropna()

        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            data=temp,
            x="AQI",
            y=target,
            alpha=0.5
        )

        plt.title(
            f"Current AQI vs {target}"
        )

        plt.xlabel("Current AQI")
        plt.ylabel(target)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / f"aqi_vs_{target}.png",
            dpi=300
        )

        plt.show()

        plt.close()


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

def outlier_analysis(df):

    columns = [
        "AQI",
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "Temperature",
        "Humidity",
        "Precipitation"
    ]

    results = []

    for column in columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        count = (
            (df[column] < lower_bound) |
            (df[column] > upper_bound)
        ).sum()

        results.append({
            "feature": column,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "potential_outliers": count
        })

    result_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 60)
    print("POTENTIAL OUTLIERS")
    print("=" * 60)

    print(
        result_df.to_string(index=False)
    )

    result_df.to_csv(
        OUTPUT_DIR / "outlier_summary.csv",
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("PEARLS AQI PREDICTOR — PHASE 2 EDA")
    print("=" * 60)

    # Load
    df = load_data()

    # Basic checks
    basic_information(df)
    missing_values(df)
    check_duplicates(df)
    date_analysis(df)
    descriptive_statistics(df)

    # AQI analysis
    plot_aqi_distribution(df)
    plot_aqi_boxplot(df)
    plot_aqi_over_time(df)
    plot_aqi_rolling_mean(df)

    # Feature distributions
    plot_pollutant_distributions(df)
    plot_weather_distributions(df)

    # Time trends
    plot_pollutants_over_time(df)

    # Relationships
    plot_aqi_vs_pollutants(df)
    plot_aqi_vs_weather(df)

    # Correlations
    aqi_correlations(df)
    plot_correlation_heatmap(df)

    # Seasonality
    monthly_analysis(df)
    day_of_week_analysis(df)
    yearly_analysis(df)

    # Lag analysis
    plot_aqi_lags(df)
    plot_aqi_change(df)

    # Targets
    target_analysis(df)

    # Outliers
    outlier_analysis(df)

    print("\n")
    print("=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)

    print(f"\nEDA outputs saved to:")
    print(OUTPUT_DIR)


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()