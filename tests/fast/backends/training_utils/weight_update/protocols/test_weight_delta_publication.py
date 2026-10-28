import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="stage-b-cpu", labels=["weight-update"])

from safetensors import safe_open

from miles.backends.training_utils.weight_update.protocols.weight_delta import publication

FRAMES = [{"offset": 0, "nbytes": 16, "inner_offset": 0, "inner_nbytes": 10}]
ZSTD = {"inner_nbytes": 10, "chunks": [{"offset": 0, "nbytes": 6, "inner_offset": 0, "inner_nbytes": 10}]}
# The same version is in SGLang's test/registered/unit/weight_sync/weight_delta/test_manifest.py.
GOLDEN_MANIFEST = (
    '{"base_version":4,"codec":"gap-zstd","files":[{"name":"owner-00000.safetensors","nbytes":1696}],'
    '"format":"weight-delta/v1","frame_nbytes":1048576,"target_version":5}'
)
GOLDEN_OWNER_HEADER = (
    '{"__metadata__":{"dtypes":"BF16,U8"},"tensors.name_end":{"dtype":"I64","shape":[2],"data_offsets":[0,16]'
    '},"tensors.shape":{"dtype":"I64","shape":[3],"data_offsets":[16,40]},"tensors.shape_end":{"dtype":"I64",'
    '"shape":[2],"data_offsets":[40,56]},"tensors.offset":{"dtype":"I64","shape":[2],"data_offsets":[56,72]},'
    '"tensors.nbytes":{"dtype":"I64","shape":[2],"data_offsets":[72,88]},"tensors.inner_nbytes":{"dtype":"I64'
    '","shape":[2],"data_offsets":[88,104]},"tensors.chunk_end":{"dtype":"I64","shape":[2],"data_offsets":[10'
    '4,120]},"tensors.frame_end":{"dtype":"I64","shape":[2],"data_offsets":[120,136]},"chunks.offset":{"dtype'
    '":"I64","shape":[1],"data_offsets":[136,144]},"chunks.nbytes":{"dtype":"I64","shape":[1],"data_offsets":'
    '[144,152]},"chunks.inner_offset":{"dtype":"I64","shape":[1],"data_offsets":[152,160]},"chunks.inner_nbyt'
    'es":{"dtype":"I64","shape":[1],"data_offsets":[160,168]},"frames.offset":{"dtype":"I64","shape":[1],"dat'
    'a_offsets":[168,176]},"frames.nbytes":{"dtype":"I64","shape":[1],"data_offsets":[176,184]},"frames.inner'
    '_offset":{"dtype":"I64","shape":[1],"data_offsets":[184,192]},"frames.inner_nbytes":{"dtype":"I64","shap'
    'e":[1],"data_offsets":[192,200]},"tensors.name":{"dtype":"U8","shape":[14],"data_offsets":[200,214]},"te'
    'nsors.dtype":{"dtype":"U8","shape":[2],"data_offsets":[214,216]},"tensors.raw":{"dtype":"BOOL","shape":['
    '2],"data_offsets":[216,218]},"payload":{"dtype":"U8","shape":[22],"data_offsets":[218,240]}}    '
)
GOLDEN_OWNER_DATA = (
    "06000000000000000e00000000000000020000000000000004000000000000000400000000000000010000000000000003000000"
    "00000000000000000000000010000000000000000400000000000000060000000000000000000000000000000a00000000000000"
    "00000000000000000100000000000000000000000000000001000000000000000000000000000000060000000000000000000000"
    "000000000a000000000000000000000000000000100000000000000000000000000000000a00000000000000612e62696173622e"
    "77656967687400010100010203040000000000000000000000007a7374642121"
)
GOLDEN_OWNER_FILE = (
    len(GOLDEN_OWNER_HEADER).to_bytes(8, "little") + GOLDEN_OWNER_HEADER.encode() + bytes.fromhex(GOLDEN_OWNER_DATA)
)


@pytest.mark.parametrize(
    "dtype, shape, raw",
    [("F32", [4, 4], True), ("BF16", [8], True), ("F32", [], True), ("BF16", [4, 4], False), ("U8", [4, 2], False)],
)
def test_fp32_tensors_and_tensors_of_at_most_one_dimension_are_raw(dtype, shape, raw):
    assert publication.is_raw(dtype, shape) == raw


def test_a_version_is_its_owner_files_and_a_manifest_that_lists_them(tmp_path):
    directory = publication.version_dir(tmp_path, 5)
    changed = publication.OwnerWriter(directory, 0)
    changed.add_raw("a.bias", "BF16", [2], b"\x01\x02\x03\x04")
    changed.add_xor("b.weight", "U8", [4, 4], FRAMES, b"zstd!!", ZSTD)
    unchanged = publication.OwnerWriter(directory, 7)

    path = publication.seal(
        directory, [unchanged.finish(), changed.finish()], codec="gap-zstd", base_version=4, target_version=5
    )

    assert path == tmp_path / "weight_v000005" / "manifest.json"
    assert path.read_text() == GOLDEN_MANIFEST
    assert sorted(file.name for file in directory.iterdir()) == ["manifest.json", "owner-00000.safetensors"]
    assert (directory / "owner-00000.safetensors").read_bytes() == GOLDEN_OWNER_FILE


def test_each_tensor_reads_back_from_its_columns_and_starts_on_the_alignment(tmp_path):
    second_frame = {"offset": 1 << 20, "nbytes": 8, "inner_offset": 16, "inner_nbytes": 3}
    two_frames = {"inner_nbytes": 19, "chunks": [{"offset": 0, "nbytes": 5, "inner_offset": 0, "inner_nbytes": 19}]}
    writer = publication.OwnerWriter(tmp_path, 0)
    writer.add_raw("a.bias", "BF16", [3], b"abcdef")
    writer.add_xor("b.weight", "U8", [4, 4], FRAMES, b"zstd!!", ZSTD)
    writer.add_xor("c.weight", "BF16", [2, 262146], [*FRAMES, second_frame], b"chunk", two_frames)
    record = writer.finish()

    path = tmp_path / "owner-00000.safetensors"
    assert record == {"name": path.name, "nbytes": path.stat().st_size}
    rows = _read_owner_file(path)
    assert rows == [
        ("a.bias", "BF16", [3], True, 0, b"abcdef", 0, [], []),
        ("b.weight", "U8", [4, 4], False, 16, b"zstd!!", 10, ZSTD["chunks"], FRAMES),
        ("c.weight", "BF16", [2, 262146], False, 32, b"chunk", 19, two_frames["chunks"], [*FRAMES, second_frame]),
    ]
    assert all(offset % publication.ALIGNMENT == 0 for _, _, _, _, offset, *_ in rows)


def test_an_owner_whose_tensors_did_not_change_writes_nothing(tmp_path):
    assert publication.OwnerWriter(tmp_path / "weight_v000001", 0).finish() is None
    assert not (tmp_path / "weight_v000001").exists()


def test_a_version_is_sealed_once(tmp_path):
    publication.seal(tmp_path, [], codec="gap-zstd", base_version=0, target_version=1)
    with pytest.raises(FileExistsError):
        publication.seal(tmp_path, [], codec="gap-zstd", base_version=0, target_version=1)


def _read_owner_file(path) -> list[tuple]:
    """Each tensor's name, dtype, shape, raw, payload offset and bytes, inner size, chunks and frames."""
    with safe_open(path, framework="numpy") as owner:
        dtypes = owner.metadata()["dtypes"].split(",")
        column = {key: owner.get_tensor(key) for key in owner.keys()}
    names, payload = bytes(column["tensors.name"]).decode(), bytes(column["payload"])

    def span(ends: str, index: int) -> slice:
        return slice(int(column[ends][index - 1]) if index else 0, int(column[ends][index]))

    def regions(table: str, rows: slice) -> list[dict]:
        fields = ("offset", "nbytes", "inner_offset", "inner_nbytes")
        return [
            {field: int(column[f"{table}.{field}"][row]) for field in fields} for row in range(rows.start, rows.stop)
        ]

    return [
        (
            names[span("tensors.name_end", index)],
            dtypes[column["tensors.dtype"][index]],
            column["tensors.shape"][span("tensors.shape_end", index)].tolist(),
            bool(column["tensors.raw"][index]),
            int(offset),
            payload[offset : offset + column["tensors.nbytes"][index]],
            int(column["tensors.inner_nbytes"][index]),
            regions("chunks", span("tensors.chunk_end", index)),
            regions("frames", span("tensors.frame_end", index)),
        )
        for index, offset in enumerate(column["tensors.offset"])
    ]
