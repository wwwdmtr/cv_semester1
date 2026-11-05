"""Загрузить только позы приседаний из фиксированного архива REHAB26.

HTTP Range позволяет не загружать остальные упражнения и RGB-видео.
Каждый извлечённый файл проверяется по CRC32 и размеру ZIP-каталога,
после чего для локальной воспроизводимости записывается SHA-256.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import ssl
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

from audit_rehab26 import MD5

RECORD_URL = "https://zenodo.org/records/23018128"
ARCHIVE_URL = RECORD_URL + "/files/jointSequences-mediapipe.zip?download=1"
ARCHIVE_SIZE = 686028292
ARCHIVE_PUBLISHED_MD5 = "303ba8e1c236ee52dac7c0b8ac23789f"
_USE_CURL = False


def curl_bytes(url, start=None, end=None):
    # На macOS Python.org иногда не настроен системный CA-store. Системный
    # curl использует свою доверенную цепочку; проверку TLS не отключаем.
    with tempfile.NamedTemporaryFile() as headers_file:
        command = ["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "90", "--dump-header", headers_file.name]
        if start is not None:
            command.extend(["--range", f"{start}-{end}"])
        command.append(url)
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode:
            raise urllib.error.URLError(result.stderr.decode("utf-8", errors="replace"))
        headers = Path(headers_file.name).read_text()
        if start is not None:
            statuses = re.findall(r"(?im)^HTTP/\S+\s+(\d+)", headers)
            ranges = re.findall(r"(?im)^content-range:\s*([^\r\n]+)", headers)
            if not statuses or statuses[-1] != "206" or not ranges or ranges[-1].strip() != f"bytes {start}-{end}/{ARCHIVE_SIZE}":
                raise ValueError("curl: сервер не подтвердил точный диапазон")
        limit = end - start + 1 if start is not None else 1024 * 1024
        if len(result.stdout) > limit or (start is not None and len(result.stdout) != limit):
            raise ValueError("curl: неверный размер ответа")
        return result.stdout


def request_bytes(url, start=None, end=None):
    global _USE_CURL
    headers = {"User-Agent": "cv-semester1-research/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    for attempt in range(5):
        try:
            if _USE_CURL:
                return curl_bytes(url, start, end)
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
                if start is not None:
                    expected_range = f"bytes {start}-{end}/{ARCHIVE_SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError(f"Сервер не подтвердил точный диапазон: {response.status}, {response.headers.get('Content-Range')}")
                limit = end - start + 1 if start is not None else 1024 * 1024
                result = response.read(limit + 1)
                if len(result) > limit or (start is not None and len(result) != limit):
                    raise ValueError("Неверный размер ответа")
                return result
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, ssl.SSLCertVerificationError):
                _USE_CURL = True
                continue
            retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in {429, 500, 502, 503, 504}
            if attempt == 4 or not retryable:
                raise
            delay = min(20, 2 ** (attempt + 1))
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    delay = min(20, max(delay, int(exc.headers.get("Retry-After", "0"))))
                except ValueError:
                    pass
            time.sleep(delay)


def zip_members():
    start = ARCHIVE_SIZE - 65536
    tail = request_bytes(ARCHIVE_URL, start, ARCHIVE_SIZE - 1)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise ValueError("Не найден завершающий ZIP-каталог")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_at)
    _, disk, directory_disk, disk_entries, total_entries, directory_size, directory_offset, comment_len = eocd
    if disk or directory_disk or disk_entries != total_entries or directory_offset + directory_size > ARCHIVE_SIZE:
        raise ValueError("Неподдерживаемый ZIP-каталог")
    if directory_offset >= start:
        directory = tail[directory_offset - start:directory_offset - start + directory_size]
    else:
        directory = request_bytes(ARCHIVE_URL, directory_offset, directory_offset + directory_size - 1)
    members, position = [], 0
    while position < len(directory):
        row = struct.unpack_from("<4s6H3I5H2I", directory, position)
        if row[0] != b"PK\x01\x02":
            raise ValueError("Повреждён центральный ZIP-каталог")
        filename = directory[position + 46:position + 46 + row[10]].decode("utf-8")
        members.append({
            "filename": filename, "method": row[4], "crc32": row[7],
            "compressed_bytes": row[8], "uncompressed_bytes": row[9], "offset": row[16],
        })
        position += 46 + row[10] + row[11] + row[12]
    if len(members) != total_entries:
        raise ValueError("Число записей ZIP не совпало")
    return members, hashlib.sha256(directory).hexdigest()


def check_file(path, member):
    if not path.exists() or path.stat().st_size != member["uncompressed_bytes"]:
        return False
    crc = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            crc = zlib.crc32(chunk, crc)
    return (crc & 0xffffffff) == member["crc32"]


def extract_member(member, output_dir):
    path = output_dir / member["filename"]
    already_present = check_file(path, member)
    if not already_present:
        begin = member["offset"]
        end = min(ARCHIVE_SIZE - 1, begin + member["compressed_bytes"] + 65535)
        block = request_bytes(ARCHIVE_URL, begin, end)
        header = struct.unpack_from("<4s5H3I2H", block, 0)
        if header[0] != b"PK\x03\x04" or header[2] & 1:
            raise ValueError(f"Неподдерживаемый локальный ZIP-заголовок: {member['filename']}")
        name = block[30:30 + header[9]].decode("utf-8")
        offset = 30 + header[9] + header[10]
        if name != member["filename"] or header[3] != member["method"]:
            raise ValueError("Локальный и центральный ZIP-заголовки не совпали")
        compressed = block[offset:offset + member["compressed_bytes"]]
        if len(compressed) != member["compressed_bytes"]:
            raise ValueError("Недостаточно сжатых данных")
        if member["method"] == 8:
            content = zlib.decompress(compressed, -15)
        elif member["method"] == 0:
            content = compressed
        else:
            raise ValueError("Неподдерживаемый метод ZIP-сжатия")
        if len(content) != member["uncompressed_bytes"] or zlib.crc32(content) & 0xffffffff != member["crc32"]:
            raise ValueError(f"Контрольная сумма не совпала: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        partial.write_bytes(content)
        partial.replace(path)
    return {
        **member,
        "crc32": f"{member['crc32']:08x}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reused_verified_file": already_present,
    }


def download(output_dir, dimensions="3d", workers=3):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_md5 in MD5.items():
        path = output_dir / filename
        if not path.exists():
            url = RECORD_URL + "/files/" + urllib.parse.quote(filename) + "?download=1"
            content = request_bytes(url)
            if hashlib.md5(content).hexdigest() != expected_md5:
                raise ValueError(f"Контрольная сумма разметки не совпала: {filename}")
            path.write_bytes(content)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected_md5:
            raise ValueError(f"Локальная разметка отличается от проверенной версии: {filename}")
    members, directory_sha = zip_members()
    suffix = "(?:2d|3d)" if dimensions == "both" else dimensions
    pattern = re.compile(rf"^mediapipe/ex6/ex6_[A-Za-z]+_cam[123]_{suffix}\.npy$")
    chosen = sorted((m for m in members if pattern.fullmatch(m["filename"])), key=lambda m: m["filename"])
    expected_count = 30 if dimensions == "both" else 15
    if len(chosen) != expected_count:
        raise ValueError(f"Ожидалось {expected_count} файлов поз, получено {len(chosen)}")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_member, member, output_dir): member for member in chosen}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(chosen)}] Проверен {result['filename']}", flush=True)
    manifest = {
        "source_record": RECORD_URL, "archive_url": ARCHIVE_URL,
        "archive_bytes": ARCHIVE_SIZE, "published_full_archive_md5": ARCHIVE_PUBLISHED_MD5,
        "full_archive_md5_verified": False,
        "validation": "Извлечены только выбранные ZIP-записи: размеры и CRC32 проверены; полный архив не скачивался.",
        "central_directory_sha256": directory_sha,
        "dimensions": dimensions, "members": sorted(results, key=lambda m: m["filename"]),
    }
    manifest_path = output_dir / f"download_manifest_{dimensions}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Готово: {len(results)} файлов; манифест {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/rehab26"))
    parser.add_argument("--dimensions", choices=("2d", "3d", "both"), default="3d")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers должен быть от 1 до 4")
    download(args.output_dir, args.dimensions, args.workers)


if __name__ == "__main__":
    main()
