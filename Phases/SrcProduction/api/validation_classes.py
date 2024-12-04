from pydantic import BaseModel, Field


class ClassifierInputFeature(BaseModel):
    AppUsageTime_min_day: int = Field(
        default=0, description="Minutes per day you use Apps in your mobile phone."
    )
    ScreenOnTime_hours_day: float = Field(
        default=0.0,
        description="Float number of hours you pass with the screen of your phone on.",
    )
    BatteryDrain_mAh_day: int = Field(default=0, description="Usage in mAh per day.")
    NumberOfAppsInstalled: int = Field(
        default=0,
        description="THe number of apps you have currently install in your phone.",
    )
    DataUsage_MB_day: int = Field(
        default=0,
        description="The usage of DataTrain you have currently with your phone",
    )
