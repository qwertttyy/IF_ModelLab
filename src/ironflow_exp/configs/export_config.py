from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExportConfig:
    output_root: str
    save_json: bool = True
    save_csv: bool = True
    save_report: bool = True
    save_bundle: bool = True
    save_visualizations: bool = True
