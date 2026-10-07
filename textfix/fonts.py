"""Bộ font designer chọn (chỉ font giấy phép OFL, đủ dấu tiếng Việt) + @font-face nhúng base64 cho đúng các font dùng."""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path



FONTS_DIR = Path(os.environ.get("TEXTFIX_FONTS", Path(__file__).resolve().parents[1] / "assets" / "fonts"))


# khoá -> (css family, lớp, fallback, {độ đậm: tệp}). Tệp nằm ở FONTS_DIR (assets/fonts/): CHỈ giấy phép OFL (dùng thương mại được), đủ 134 ký tự dấu tiếng Việt (02/10: bỏ mọi SVN-*).
# Bản nghiêng THẬT là họ riêng ('... Italic'), không dùng nghiêng giả của Chromium.
CATALOG = {
    # không chân
    "bevietnam": ("Be Vietnam Pro", "sans", "sans-serif",
                  {400: "BeVietnamPro-Regular.woff2", 600: "BeVietnamPro-SemiBold.woff2", 700: "BeVietnamPro-Bold.woff2",
                   800: "BeVietnamPro-ExtraBold.woff2", 900: "BeVietnamPro-Black.woff2"}),
    "montserrat": ("Montserrat", "sans", "sans-serif", {"100 900": "Montserrat.woff2"}),
    "inter": ("Inter", "sans", "sans-serif", {"100 900": "Inter.woff2"}),
    "jakarta": ("Plus Jakarta Sans", "sans", "sans-serif", {"200 800": "PlusJakartaSans.woff2"}),
    "lexend": ("Lexend", "sans", "sans-serif", {"100 900": "Lexend.woff2"}),
    "archivo": ("Archivo", "sans", "sans-serif", {"100 900": "Archivo.woff2"}),
    "chakra": ("Chakra Petch", "sans", "sans-serif", {700: "ChakraPetch-Bold.woff2"}),
    # hẹp
    "oswald": ("Oswald", "sans", "sans-serif", {"200 700": "Oswald.woff2"}),
    "anton": ("Anton", "sans", "sans-serif", {400: "Anton-Regular.woff2"}),
    "barlowcond": ("Barlow Condensed", "sans", "sans-serif",
                   {800: "BarlowCondensed-ExtraBold.woff2", 900: "BarlowCondensed-Black.woff2"}),
    "sairaxc": ("Saira ExtraCondensed", "sans", "sans-serif",
                {700: "SairaExtraCondensed-Bold.woff2", 900: "SairaExtraCondensed-Black.woff2"}),
    "bigshoulders": ("Big Shoulders Display", "sans", "sans-serif", {"100 900": "BigShouldersDisplay.woff2"}),
    "robotocond": ("Roboto Condensed", "sans", "sans-serif", {"100 900": "RobotoCondensed.woff2"}),
    # không chân NGHIÊNG (thể thao / tốc độ)
    "bevietnami": ("Be Vietnam Pro Italic", "sans", "sans-serif",
                   {700: "BeVietnamPro-BoldItalic.woff2", 800: "BeVietnamPro-ExtraBoldItalic.woff2",
                    900: "BeVietnamPro-BlackItalic.woff2"}),
    "montserrati": ("Montserrat Italic", "sans", "sans-serif", {"100 900": "Montserrat-Italic.woff2"}),
    "kaniti": ("Kanit Italic", "sans", "sans-serif", {800: "Kanit-ExtraBoldItalic.woff2", 900: "Kanit-BlackItalic.woff2"}),
    "barlowcondi": ("Barlow Condensed Italic", "sans", "sans-serif", {900: "BarlowCondensed-BlackItalic.woff2"}),
    "sairai": ("Saira Italic", "sans", "sans-serif", {"100 900": "Saira-Italic.woff2"}),
    "chakrai": ("Chakra Petch Italic", "sans", "sans-serif", {700: "ChakraPetch-BoldItalic.woff2"}),
    # tròn / vui
    "baloo": ("Baloo 2", "sans", "sans-serif", {"400 800": "Baloo2.woff2"}),
    "grandstander": ("Grandstander", "sans", "sans-serif", {"100 900": "Grandstander.woff2"}),
    "bungee": ("Bungee", "sans", "sans-serif", {400: "Bungee-Regular.woff2"}),
    "paytone": ("Paytone One", "sans", "sans-serif", {400: "PaytoneOne-Regular.woff2"}),
    # có chân
    "playfair": ("Playfair Display", "serif", "serif", {"400 900": "PlayfairDisplay.woff2"}),
    "cormorant": ("Cormorant", "serif", "serif", {"300 700": "Cormorant.woff2"}),
    "lora": ("Lora", "serif", "serif", {"400 700": "Lora.woff2"}),
    "prata": ("Prata", "serif", "serif", {400: "Prata-Regular.woff2"}),
    "yeseva": ("Yeseva One", "serif", "serif", {400: "YesevaOne-Regular.woff2"}),
    "ebgaramond": ("EB Garamond", "serif", "serif", {"400 800": "EBGaramond.woff2"}),
    "fraunces": ("Fraunces", "serif", "serif", {"100 900": "Fraunces.woff2"}),
    "playfairi": ("Playfair Display Italic", "serif", "serif", {"400 900": "PlayfairDisplay-Italic.woff2"}),
    "cormoranti": ("Cormorant Italic", "serif", "serif", {"300 700": "Cormorant-Italic.woff2"}),
    "lorai": ("Lora Italic", "serif", "serif", {"400 700": "Lora-Italic.woff2"}),
    # thư pháp
    "dancing": ("Dancing Script", "script", "cursive", {"400 700": "DancingScript.woff2"}),
    "greatvibes": ("Great Vibes", "script", "cursive", {400: "GreatVibes-Regular.woff2"}),
    "alexbrush": ("Alex Brush", "script", "cursive", {400: "AlexBrush-Regular.woff2"}),
    "pacifico": ("Pacifico", "script", "cursive", {400: "Pacifico-Regular.woff2"}),
    "allura": ("Allura", "script", "cursive", {400: "Allura-Regular.woff2"}),
    "imperial": ("Imperial Script", "script", "cursive", {400: "ImperialScript-Regular.woff2"}),
    "playball": ("Playball", "script", "cursive", {400: "Playball-Regular.woff2"}),
    "lobster": ("Lobster", "script", "cursive", {400: "Lobster-Regular.woff2"}),
    # viết tay thường / cọ
    "mali": ("Mali", "script", "cursive", {700: "Mali-Bold.woff2"}),
    "patrickhand": ("Patrick Hand", "script", "cursive", {400: "PatrickHand-Regular.woff2"}),
    "sriracha": ("Sriracha", "script", "cursive", {400: "Sriracha-Regular.woff2"}),
    "mansalva": ("Mansalva", "script", "cursive", {400: "Mansalva-Regular.woff2"}),
    "sedgwick": ("Sedgwick Ave Display", "script", "cursive", {400: "SedgwickAveDisplay-Regular.woff2"}),
}


def family(key: str) -> str:
    fam, _, fb, _ = CATALOG[key]
    return f"'{fam}', {fb}"


@lru_cache(maxsize=64)
def faces_for(keys: tuple) -> str:
    """@font-face (nhúng base64) cho đúng các khoá CATALOG cần."""
    out = []
    for key in keys:
        fam, _, _, files = CATALOG[key]
        for wt, fn in files.items():
            p = FONTS_DIR / fn
            fmt = "woff2" if fn.endswith(".woff2") else "truetype"
            b64 = base64.b64encode(p.read_bytes()).decode()
            out.append(f"@font-face{{font-family:'{fam}';src:url('data:font/{fmt};base64,{b64}') format('{fmt}');"
                       f"font-weight:{wt};font-style:normal;unicode-range:U+0-9F,U+A1-10FFFF}}")
    return "\n".join(out)
