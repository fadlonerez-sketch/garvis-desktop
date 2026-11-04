import pytest

from app.tools.filesystem import ListDirectoryTool, ReadTextFileTool


@pytest.mark.asyncio
async def test_read_text_file_within_root(tmp_path):
    (tmp_path / "note.txt").write_text("hello garvis", encoding="utf-8")
    tool = ReadTextFileTool(root=tmp_path)
    args = tool.validate_arguments({"relative_path": "note.txt"})
    result = await tool.execute(args)
    assert result.success
    assert result.data["content"] == "hello garvis"
    assert result.data["total_lines"] == 1


@pytest.mark.asyncio
async def test_read_text_file_rejects_path_traversal(tmp_path):
    tool = ReadTextFileTool(root=tmp_path)
    args = tool.validate_arguments({"relative_path": "../../secret.txt"})
    result = await tool.execute(args)
    assert not result.success
    assert "escapes" in result.error.lower()


@pytest.mark.asyncio
async def test_list_directory(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    tool = ListDirectoryTool(root=tmp_path)
    args = tool.validate_arguments({})
    result = await tool.execute(args)
    assert result.success
    names = {entry["name"] for entry in result.data}
    assert names == {"a.txt", "sub"}
