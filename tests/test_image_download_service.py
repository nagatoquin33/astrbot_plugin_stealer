import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from astrbot_plugin_stealer.core.events.image_download_service import (
    ImageDownloadService,
)


class ImageDownloadServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_event_image_is_copied_before_temp_processing(self):
        async def run_inline(func, *args, **kwargs):
            return func(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "event-owned.png"
            source_content = b"\x89PNG\r\n\x1a\n" + b"event-owned-image"
            source_path.write_bytes(source_content)
            image = types.SimpleNamespace(
                path=str(source_path),
                file=str(source_path),
                url=str(source_path),
            )

            service = ImageDownloadService()
            with mock.patch(
                "astrbot_plugin_stealer.core.events.image_download_service.asyncio.to_thread",
                new=run_inline,
            ):
                plugin_temp_path, is_gif = await service.download_original_image(image)

            self.assertIsNotNone(plugin_temp_path)
            plugin_temp = Path(plugin_temp_path)
            self.assertNotEqual(plugin_temp.resolve(), source_path.resolve())
            self.assertEqual(plugin_temp.suffix, ".png")
            self.assertEqual(plugin_temp.read_bytes(), source_content)
            self.assertFalse(is_gif)

            moved_path = Path(temp_dir) / "plugin-owned.png"
            shutil.move(plugin_temp, moved_path)

            self.assertTrue(source_path.exists())
            self.assertEqual(source_path.read_bytes(), source_content)


if __name__ == "__main__":
    unittest.main()
