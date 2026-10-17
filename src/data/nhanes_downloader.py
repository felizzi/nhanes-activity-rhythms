"""
CDC NHANES Data Downloader and Processor.
Retrieves and caches SAS Transport (.XPT) files directly from the CDC NHANES portal.
"""

import os
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import pyreadstat

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class NHANESDownloader:
    """
    Downloads and caches CDC NHANES survey data (.XPT format) across multiple cycles.
    """

    def __init__(self, raw_data_dir: Optional[Path] = None):
        self.raw_data_dir = Path(raw_data_dir or config.RAW_DATA_DIR)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def get_component_filename(self, component: str, cycle: str) -> str:
        """
        Computes the standard CDC filename for a component in a specific cycle.
        E.g., for BMX in 2017-2018 -> BMX_J.XPT, in 1999-2000 -> BMX.XPT.
        """
        suffix = config.SURVEY_CYCLES.get(cycle, {}).get("suffix", "")
        return f"{component}{suffix}.XPT"

    def download_xpt(self, component: str, cycle: str, force_download: bool = False) -> Path:
        """
        Downloads a single XPT file for a given component and survey cycle.
        """
        filename = self.get_component_filename(component, cycle)
        cycle_dir = self.raw_data_dir / cycle
        cycle_dir.mkdir(parents=True, exist_ok=True)
        dest_path = cycle_dir / filename

        if dest_path.exists() and not force_download and dest_path.stat().st_size > 0:
            logger.info("Using cached file: %s", dest_path)
            return dest_path

        url = f"{config.CDC_NHANES_BASE_URL}/{cycle}/{filename}"
        logger.info("Downloading %s from %s ...", filename, url)

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
                chunk = response.read(1024 * 1024)
                while chunk:
                    out_file.write(chunk)
                    chunk = response.read(1024 * 1024)
            logger.info("Successfully downloaded: %s (%d bytes)", filename, dest_path.stat().st_size)
            return dest_path
        except Exception as err:
            if dest_path.exists():
                dest_path.unlink()
            logger.error("Failed to download %s: %s", url, err)
            raise

    def load_xpt_dataframe(self, file_path: Path) -> pd.DataFrame:
        """
        Reads a SAS XPT file into a pandas DataFrame using pyreadstat.
        """
        df, _ = pyreadstat.read_xport(str(file_path))
        return df

    def fetch_cycle_dataset(
        self,
        cycle: str,
        components: Optional[List[str]] = None,
        variable_mappings: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Downloads and merges multiple components for a cycle on SEQN (respondent ID).
        Components typically: ['DEMO', 'BMX', 'BPX', 'GHB', 'DR1TOT'].
        """
        if components is None:
            components = ["DEMO", "BMX", "GHB"]

        merged_df = None
        for comp in components:
            try:
                xpt_path = self.download_xpt(comp, cycle)
                df = self.load_xpt_dataframe(xpt_path)
                if "SEQN" not in df.columns:
                    logger.warning("SEQN not found in %s, skipping", comp)
                    continue

                if merged_df is None:
                    merged_df = df
                else:
                    cols_to_use = [c for c in df.columns if c == "SEQN" or c not in merged_df.columns]
                    merged_df = pd.merge(merged_df, df[cols_to_use], on="SEQN", how="inner")
            except Exception as e:
                logger.warning("Could not fetch component %s for cycle %s: %s", comp, cycle, e)

        if merged_df is None:
            return pd.DataFrame()

        merged_df["cycle"] = cycle
        merged_df["survey_year"] = config.SURVEY_CYCLES.get(cycle, {}).get("year", 2000)

        mappings = variable_mappings or config.VARIABLE_MAPPINGS
        rename_dict = {orig: new for orig, new in mappings.items() if orig in merged_df.columns}
        merged_df = merged_df.rename(columns=rename_dict)
        return merged_df
