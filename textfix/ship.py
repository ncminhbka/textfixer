"""MANG KẾT QUẢ VỀ từ máy chủ JupyterLab: gói các tệp thành nhiều zip ĐỘC LẬP, mỗi zip < 27 MB (giải nén từng cái, không phải
ghép), rồi tự tải về trình duyệt.

  zips = pack(files, out_dir, "pairs")   # files: [(đường dẫn trên đĩa, tên trong zip)] -> [out_dir/pairs_01.zip, ...]
  offer(zips)                            # trong ô notebook (%run scripts/...): tự bấm tải từng zip + hiện link dự phòng;
                                         # ngoài notebook (terminal): in đường dẫn để tải tay từ cây thư mục JupyterLab

Tự tải cần chạy script BẰNG Ô NOTEBOOK (`%run scripts/<tên>.py ...`): chỉ khi đó code mới gửi được JavaScript tới trình duyệt.
Trình duyệt có thể hỏi "cho phép tải nhiều tệp" lần đầu: chọn cho phép.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

LIMIT = 27 * 1024 * 1024
HEADROOM = 256 * 1024   # chừa cho mục lục zip / tên tệp


def pack(files: list[tuple[Path, str]], out_dir: Path, name: str, limit: int = LIMIT) -> list[Path]:
    """Xếp tệp vào các zip độc lập < limit (xếp tham lam theo cỡ giảm dần vào zip đầu tiên còn chỗ). Ảnh đã nén (png / jpg) lưu
    thẳng, còn lại deflate. Tệp đơn lẻ to hơn limit -> lỗi (cắt nhỏ tệp trước khi gói)."""
    cap = limit - HEADROOM
    items = sorted(((Path(p), arc, Path(p).stat().st_size) for p, arc in files), key=lambda t: -t[2])
    big = [str(p) for p, _, s in items if s > cap]
    if big:
        raise ValueError(f"tệp lớn hơn {cap / 2**20:.1f} MB, không gói được: {big}")
    bins: list[list] = []
    for p, arc, s in items:
        b = next((b for b in bins if b[0] + s <= cap), None)
        if b is None:
            b = [0, []]
            bins.append(b)
        b[0] += s
        b[1].append((p, arc))
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{name}_*.zip"):
        old.unlink()
    zips = []
    for k, (_, members) in enumerate(bins, 1):
        z = out_dir / f"{name}_{k:02d}.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for p, arc in sorted(members, key=lambda m: m[1]):
                stored = p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".zip")
                zf.write(p, arc, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
        if z.stat().st_size >= limit:   # không xảy ra khi HEADROOM đủ; chặn cứng để không bao giờ gửi tệp quá cỡ
            raise RuntimeError(f"{z} = {z.stat().st_size} B >= {limit} B")
        zips.append(z)
    return zips


def pack_dir(src: Path, out_dir: Path, name: str, pattern: str = "*", limit: int = LIMIT) -> list[Path]:
    """Gói mọi tệp khớp pattern trong src (đệ quy), tên trong zip = đường dẫn tương đối từ src."""
    files = [(p, str(p.relative_to(src)).replace("\\", "/")) for p in sorted(Path(src).rglob(pattern)) if p.is_file()]
    return pack(files, out_dir, name, limit)


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and type(ip).__name__ == "ZMQInteractiveShell"
    except ImportError:
        return False


_JS = r"""
(function (paths, box) {
  // đường dẫn tuyệt đối -> URL /files/ của Jupyter: baseUrl (JupyterHub: /user/<tên>/) và thư mục gốc máy chủ lấy từ cấu hình
  // trang JupyterLab; thiếu thì coi gốc = HOME
  var cfg = {};
  try { cfg = JSON.parse(document.getElementById('jupyter-config-data').textContent); } catch (e) {}
  var base = (cfg.baseUrl || '/').replace(/\/?$/, '/');
  var root = (cfg.serverRoot || '%HOME%').replace(/^~/, '%HOME%').replace(/\/?$/, '/');
  var ul = document.getElementById(box);
  paths.forEach(function (p, i) {
    var rel = p.indexOf(root) === 0 ? p.slice(root.length) : p.replace(/^\//, '');
    var url = base + 'files/' + rel.split('/').map(encodeURIComponent).join('/');
    var name = p.split('/').pop();
    if (ul) { var li = document.createElement('li'), a = document.createElement('a');
              a.href = url; a.download = name; a.textContent = name; li.appendChild(a); ul.appendChild(li); }
    setTimeout(function () {
      var a = document.createElement('a');
      a.href = url; a.download = name;
      document.body.appendChild(a); a.click(); a.remove();
    }, 1200 * i);
  });
})(%PATHS%, '%BOX%');
"""


def offer(zips: list[Path]) -> None:
    """Tự tải các zip về (trong notebook), kèm link dự phòng; ngoài notebook in đường dẫn."""
    import os
    import uuid
    zips = [Path(z).resolve() for z in zips]
    total = sum(z.stat().st_size for z in zips) / 2**20
    print(f"đã gói {len(zips)} zip ({total:.1f} MB), mỗi zip < {LIMIT / 2**20:.0f} MB:")
    for z in zips:
        print(f"  {z}  ({z.stat().st_size / 2**20:.1f} MB)")
    if not _in_notebook():
        print("không chạy trong notebook -> tải tay từ cây thư mục JupyterLab (chuột phải -> Download), hoặc chạy lại bằng "
              "ô notebook: %run scripts/<tên>.py ... để tự tải")
        return
    from IPython.display import HTML, Javascript, display
    box = "ship-" + uuid.uuid4().hex[:8]
    home = os.path.expanduser("~").replace("\\", "/")
    paths = [str(z).replace("\\", "/") for z in zips]
    display(HTML(f"<p>Đang tự tải {len(paths)} zip. Không thấy tải thì bấm từng link:</p><ul id='{box}'></ul>"))
    display(Javascript(_JS.replace("%HOME%", home).replace("%PATHS%", json.dumps(paths)).replace("%BOX%", box)))
