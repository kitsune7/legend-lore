"""Unit tests for YouTube input handling. No network: subprocess/yt-dlp are mocked."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dnd_notetaker import youtube

VID = "dQw4w9WgXcQ"


class VideoIdTests(unittest.TestCase):
    def test_supported_forms(self):
        urls = [
            f"https://www.youtube.com/watch?v={VID}",
            f"https://youtube.com/watch?v={VID}&t=90s",
            f"https://www.youtube.com/watch?list=PLxyz&v={VID}",
            f"https://m.youtube.com/watch?v={VID}",
            f"https://music.youtube.com/watch?v={VID}",
            f"https://youtu.be/{VID}",
            f"https://youtu.be/{VID}?t=10",
            f"https://youtu.be/{VID}/",
            f"https://www.youtube.com/shorts/{VID}",
            f"https://www.youtube.com/live/{VID}?feature=share",
            f"https://www.youtube.com/embed/{VID}",
        ]
        for u in urls:
            with self.subTest(url=u):
                self.assertEqual(youtube.video_id(u), VID)

    def test_rejected(self):
        for s in [
            "data/session.wav",
            VID,
            "https://www.youtube.com/playlist?list=PLxyz",
            "https://www.youtube.com/@somechannel",
            "https://example.com/watch?v=" + VID,
            "https://www.youtube.com/watch?v=" + VID[:10],
            "https://www.youtube.com/watch?v=" + VID + "x",
            f"ftp://youtu.be/{VID}",
        ]:
            with self.subTest(s=s):
                self.assertIsNone(youtube.video_id(s))

    def test_host_detection_separates_unsupported_youtube_urls_from_files(self):
        self.assertTrue(youtube.is_youtube_host("https://www.youtube.com/playlist?list=x"))
        self.assertFalse(youtube.is_youtube_host("session.m4a"))


class CheckAvailableTests(unittest.TestCase):
    def _run(self, info=None, exc=None):
        ydl = mock.MagicMock()
        ydl.__enter__.return_value = ydl
        if exc:
            ydl.extract_info.side_effect = exc
        else:
            ydl.extract_info.return_value = info
        with mock.patch("yt_dlp.YoutubeDL", return_value=ydl):
            return youtube.check_available("https://youtu.be/" + VID)

    def test_live_and_upcoming_rejected(self):
        for status in ("is_live", "is_upcoming"):
            with self.subTest(status=status), self.assertRaises(youtube.YoutubeError):
                self._run({"live_status": status, "formats": [{}]})

    def test_download_error_wrapped(self):
        from yt_dlp.utils import DownloadError

        with self.assertRaisesRegex(youtube.YoutubeError, "Private video"):
            self._run(exc=DownloadError("Private video"))

    def test_finished_vod_ok(self):
        info = {"live_status": "was_live", "formats": [{}]}
        self.assertEqual(self._run(info), info)


class FakeProc:
    def __init__(self, rc, stderr=b"", stdout=None):
        self.returncode = rc
        self.stderr = io.BytesIO(stderr)
        self.stdout = stdout or mock.MagicMock()
        self._rc = rc

    def wait(self):
        return self._rc

    def poll(self):
        return self._rc

    def terminate(self):
        pass


class StreamTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dst = Path(tmp.name) / "out_16k.wav"
        p = mock.patch.object(youtube, "check_available", return_value={})
        p.start()
        self.addCleanup(p.stop)

    def _stream(self, ytdlp, ffmpeg):
        calls = []

        def popen(cmd, **kw):
            calls.append((cmd, kw))
            proc = ytdlp if len(calls) == 1 else ffmpeg
            if len(calls) == 2:
                self.dst.write_bytes(b"RIFF")  # ffmpeg creates the file
            return proc

        with mock.patch("subprocess.Popen", side_effect=popen):
            try:
                return youtube.stream_to_16k_mono("https://youtu.be/" + VID, self.dst), calls
            finally:
                self.calls = calls

    def test_success_wires_ytdlp_stdout_into_ffmpeg(self):
        yt = FakeProc(0)
        out, calls = self._stream(yt, FakeProc(0))
        self.assertEqual(out, self.dst)
        self.assertTrue(self.dst.exists())
        ff_cmd, ff_kw = calls[1]
        self.assertIs(ff_kw["stdin"], yt.stdout)
        for flag in (["-i", "pipe:0"], ["-ac", "1"], ["-ar", "16000"]):
            i = ff_cmd.index(flag[0])
            self.assertEqual(ff_cmd[i + 1], flag[1])
        yt_cmd = calls[0][0]
        self.assertIn("youtube:player_client=ios,android,tv", yt_cmd)
        self.assertEqual(yt_cmd[yt_cmd.index("-o") + 1], "-")

    def test_ytdlp_failure_raises_with_stderr_and_removes_partial(self):
        with self.assertRaises(youtube.YoutubeError) as cm:
            self._stream(FakeProc(1, b"HTTP Error 403"), FakeProc(0))
        self.assertIn("yt-dlp exited 1", str(cm.exception))
        self.assertIn("HTTP Error 403", str(cm.exception))
        self.assertFalse(self.dst.exists())

    def test_ffmpeg_failure_raises_with_stderr_and_removes_partial(self):
        with self.assertRaises(youtube.YoutubeError) as cm:
            self._stream(FakeProc(0), FakeProc(1, b"Invalid data"))
        self.assertIn("ffmpeg exited 1", str(cm.exception))
        self.assertIn("Invalid data", str(cm.exception))
        self.assertFalse(self.dst.exists())

    def test_interrupt_removes_partial(self):
        class Boom(FakeProc):
            def __init__(self, rc):
                super().__init__(rc)
                self.interrupted = False

            def wait(self):
                if not self.interrupted:
                    self.interrupted = True
                    raise KeyboardInterrupt
                return self._rc

            def poll(self):
                return None

        with self.assertRaises(KeyboardInterrupt):
            self._stream(FakeProc(0), Boom(0))
        self.assertFalse(self.dst.exists())


class PipelineBranchTests(unittest.TestCase):
    def test_url_uses_video_id_stem_and_keeps_url_in_json(self):
        import json

        from dnd_notetaker import pipeline

        url = f"https://youtu.be/{VID}"
        with tempfile.TemporaryDirectory() as d:
            wav = Path(d) / f"{VID}_16k.wav"
            with mock.patch.object(pipeline.youtube, "stream_to_16k_mono", return_value=wav) as s, \
                 mock.patch.object(pipeline.preprocess, "to_16k_mono") as local, \
                 mock.patch.object(pipeline.asr, "transcribe", return_value=[]), \
                 mock.patch.object(pipeline.diarize, "diarize", return_value=[]), \
                 mock.patch.object(pipeline.merge, "merge", return_value=[]):
                res = pipeline.run(url, d)
            s.assert_called_once_with(url, Path(d) / f"{VID}_16k.wav")
            local.assert_not_called()
            self.assertTrue(res["json"].endswith(f"{VID}.transcript.json"))
            self.assertEqual(json.loads(Path(res["json"]).read_text())["audio"], url)

    def test_unsupported_youtube_url_is_rejected_not_sent_to_ffmpeg(self):
        from dnd_notetaker import pipeline

        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(pipeline.preprocess, "to_16k_mono") as local:
            with self.assertRaisesRegex(ValueError, "unsupported YouTube URL"):
                pipeline.run("https://www.youtube.com/playlist?list=PLx", d)
            local.assert_not_called()


if __name__ == "__main__":
    unittest.main()
