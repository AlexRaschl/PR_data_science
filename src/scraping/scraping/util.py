import re

DEBUG = False


class VideoInspector:
    v_id_regex = re.compile(r'href="/watch\?v=([A-Za-z0-9_\-]{11})"', re.IGNORECASE)
    title_regex = re.compile(r'title="([\s\S]*?)"')
    descr_regex = re.compile(r' dir="ltr">([\s\S]*)</div>')

    def extract(self, vid_a, vid_descr) -> dict:

        v_id = self.v_id_regex.search(vid_a).group(1)
        title = self.title_regex.search(vid_a).group(1)
        descr = self.descr_regex.search(vid_descr).group(1)

        if DEBUG:
            print(f'v_id={v_id}\n')
            print(f'title={title}\n')
            print(f'descr={descr}\n')

        return {
            'v_id': v_id,
            'title': title,
            'descr': descr
        }

    def is_music_video(self, video_info: dict, song_name: str, creator: str) -> bool:

        official_vid_regex = re.compile(r'^(un)official[\s]*vid(eo)?|^(un)official[\s]*music[\s]*vid(eo)?',
                                        re.IGNORECASE)

        # TODO: Hard fact -> Song title in video title
        # TODO: Hard fact -> 'Lyrics' not in video title
        # TODO: Hard fact -> 'Fan-vid(eo)' not in video description
        # TODO: Soft hint -> video in title
        # TODO: Soft hint -> creator name in channel name
        # TODO: Soft hint -> Official music video in description
        # TODO: Hard Fact -> Soundtrack not in video title

        title = video_info['title']
        descr = video_info['descr']
        # Hard fact -> Cover not in title
        if 'cover' in title.lower():
            print(f'Title: {video_info["title"]} : contains cover! ')
            return False

        m = official_vid_regex.search(title)
        if m is not None:
            print(f'offical_Regmatch_title: {title}')
            print(m.group()[0])
            return True

        if official_vid_regex.search(descr):
            print(f'offical_Regmatch_descr: {descr}')
            return True

        return False
