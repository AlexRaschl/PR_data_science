import re
import traceback


DEBUG = False


class VideoInspector:
    v_id_regex = re.compile(r'href="/watch\?v=([A-Za-z0-9_\-]{11})"', re.IGNORECASE)
    # title_regex = re.compile(r'title=((["\']([\s\S]*?)["\'])|(\'([\s\S]*?)\'))')
    title_regex = re.compile(r'dir="ltr">([\s\S]*?)</a>')
    descr_regex = re.compile(r' dir="ltr">([\s\S]*)</div>')
    view_regex = re.compile(r'<li>([,.0-9]*) views</li></ul>')

    official_vid_regex = re.compile(
        r'((?<!un)(official|officiell)[/a-z<>\s]*vid(eo)?)|((?<!un)(official|officiell)[a-z/<>\s]*((music[a-z/<>\s]*vid(eo)?)|MV))',
        re.IGNORECASE)
    directed_by_regex = re.compile(r'directed[\s-]*by', re.IGNORECASE)
    full_video_regex = re.compile(r'full[/a-z<>\s]*vid(eo)?')

    def extract(self, vid_a, vid_descr, vid_meta, vid_channel) -> dict:
        try:
            v_id = self.v_id_regex.search(vid_a).group(1)
            title = self.title_regex.search(vid_a).group(1)
            descr = self.descr_regex.search(vid_descr).group(1)
            views = self.view_regex.search(vid_meta).group(1).replace(',', '')

            if DEBUG:
                print(f'v_id={v_id}\n')
                print(f'title={title}\n')
                print(f'descr={descr}\n')
                print(f'views={views}\n')
                print(f'channel={vid_channel}\n')

            return {
                'v_id': v_id,
                'title': title,
                'descr': descr,
                'views': views,
                'channel': vid_channel
            }
        except Exception as e:
            traceback.print_exc()
            with open('error.log', 'a') as f:
                print(traceback.format_exc(), file=f)
                print(vid_a, file=f)
                print(vid_descr, file=f)
                print('\n\n\n\n\n\n\n\n\n', file=f)

    def is_music_video(self, video_info: dict, song_name: str, creator: str) -> bool:

        # TODO: Hard fact -> Song title in video title
        # TODO: Hard fact -> 'Lyrics' not in video title
        # TODO: Hard fact -> 'Fan-vid(eo)' not in video description
        # TODO: Soft hint -> video in title
        # TODO: Soft hint -> creator name in channel name
        # DONE: Hard fact -> Official music video in description
        # TODO: Hard Fact -> Soundtrack not in video title
        # DONE: Hard fact -> Find directed by in descr

        # TODO: EXACT MATCH OF Creator - Title counts as MV

        title = video_info['title']
        descr = video_info['descr']
        channel = video_info['channel']

        # Hard fact -> Cover not in title
        if 'cover' in title.lower() or 'lyric' in title.lower():
            print(f'Title: {title} : contains cover or lyrics! ')
            return False

        if creator.lower() not in title.lower() and creator.lower() not in channel.lower():
            print(f'Creator not in channel nor in title: {creator}')
            return False

        m = self.official_vid_regex.search(title)
        if m is not None:
            print(f'Official Video in Title: {title}')
            return True

        if self.full_video_regex.search(title):
            print(f'Full Video in title: {title}')
            return True

        if self.official_vid_regex.search(descr):
            print(f'Official Video in Description: {descr}')
            return True

        if self.directed_by_regex.search(title) is not None:
            print(f'Directed by regex matched: {descr}')
            return True

        exact_match_regex = re.compile(f'{creator} - {song_name}', re.IGNORECASE)
        if exact_match_regex.match(title):
            print(f'Exact Match Regex matched: {title}')
            return True

        return False
