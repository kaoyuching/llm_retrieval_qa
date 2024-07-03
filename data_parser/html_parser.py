import os
from typing import Optional, List, Dict
from io import BytesIO
import requests
import re
import json
import copy
from urllib.parse import urljoin
import bs4
import numpy as np
from PIL import Image


def remove_attrs(soup, preserve_attrs: Optional[List] = None):
    if soup is None:
        return soup

    new_attrs = {}
    if preserve_attrs is not None:
        for k in preserve_attrs:
            if soup.attrs.get(k, None):
                new_attrs[k] = soup.attrs.get(k)
    soup.attrs = {**new_attrs}
    return soup


def modify_tags(
    soup,
    replace_tags: Optional[Dict[str, str]] = None,
    decompose_tags: Optional[List[str]] = None,
    preserve_tags: Optional[List[str]] = None,
    preserve_attrs: Optional[List[str]] = None,
):
    r'''
        - replace_tags: {rule: new_tag_name}
        - decompose_tags: tags want to remove
        - preserve_tags: tags want to preserve
        - preserve_attrs: attrs want to preserve
    '''
    # replace tags with given rule
    if replace_tags and len(replace_tags) > 0:
        for rule, new_tag_name in replace_tags.items():
            res = soup.select(rule)
            if len(res) == 0:
                continue
            else:
                for x in res:
                    x.name = new_tag_name

    for content in soup.find_all():
        if isinstance(content, bs4.element.NavigableString):
            continue
        else:
            tag_name = content.name
            if tag_name is None:
                continue
            elif decompose_tags and any([re.search(x, tag_name) is not None for x in decompose_tags]):
                content = content.decompose()
            elif preserve_tags and (tag_name in preserve_tags):
                _ = remove_attrs(content, preserve_attrs=preserve_attrs)
            else:
                _ = content.unwrap()
    return soup


def iter_strings(elem):
    # iterate strings so that they can be replaced
    iter = elem.strings
    n = next(iter, None)
    while n is not None:
        current = n
        n = next(iter, None)
        yield current

def replace_strings(element, pattern):
    # replace all found `substring`'s with newstring
    for string in iter_strings(element):
        new_str = re.sub(pattern, ' ', string)
        string.replace_with(new_str)


class DataFromUrl():
    r"""
        - url: where to crawl data
        - encoding: data encoding
        - filter_tags: filter data with given tags
    """
    def __init__(self, url: str, encoding: str = 'utf8', filter_tags: Optional[List[str]] = None):
        self.encoding = encoding
        self.resp = requests.get(url)
        self.text = self.resp.content.decode(self.encoding)
        self.soup = bs4.BeautifulSoup(self.text, features="html.parser")
        self.filter_tags = filter_tags

    def get_data(
        self,
        replace_tags: Optional[Dict[str, str]] = None,
        decompose_tags: Optional[List[str]] = None,
        preserve_tags: Optional[List[str]] = None,
        preserve_attrs: Optional[List[str]] = None,
        save_filename: Optional[str] = None, 
    ):
        html_list = []
        for item in self.soup.find_all(self.filter_tags):
            _ = modify_tags(
                item,
                replace_tags=replace_tags,
                decompose_tags=decompose_tags,
                preserve_tags=preserve_tags,
                preserve_attrs=preserve_attrs,
            )

            body_contents = ''.join([str(x) for x in item.contents])
            html_list.append(body_contents.replace(u'\xa0', u' '))
        html_list = [x.replace(u'\xa0', u' ') for x in html_list]
        html_list, image_src = self._postprocess(html_list)
        
        if save_filename is not None:
            dirname = os.path.dirname(save_filename)
            os.makedirs(dirname, exist_ok=True)
            new_html = ''.join(html_list)
            with open(save_filename, 'w', encoding=self.encoding) as f:
                f.write(new_html)
        return html_list, image_src

    def _postprocess(self, html_list: List[str]):
        new_html_list = []
        image_src = {}

        for i, content in enumerate(html_list):
            content = re.sub(r' +', ' ', content)
            content = re.sub(r'\r', '', content)
            content = re.sub(r'\n+', '\n', content)
            content = re.sub(r' *\n *', '\n', content)
            content = re.sub(r'(\n )+', '\n', content)

            soup = bs4.BeautifulSoup(content, "html.parser")
            for _content in soup.find_all():
                if _content.name is None:
                    continue
                if re.match('^h.', _content.name):
                    if _content.get_text().replace('\n', '') == '':
                        _content.decompose()
                    replace_strings(_content, '(?<=\w|,|;|&)\n')
                if _content.name == 'a':
                    if _content.get_text().replace('\n', '') == '':
                        _content.decompose()
                if _content.name == 'p':
                    if _content.get_text().replace('\n', '') == '':
                        _content.decompose()
                    replace_strings(_content, '(?<=\w|,|;|&)\n')
                if _content.name == 'img':
                    img_src = _content.attrs['src']
                    img_url = urljoin(url, img_src)

                    response = requests.get(img_url)
                    img = Image.open(BytesIO(response.content))
                    image_src[img_src] = img
            new_html_list.append(str(soup))
        return new_html_list, image_src

    def save_image(self, image_src: Dict[str, np.ndarray]):
        # save image
        for path, img in image_src.items():
            dirname = os.path.dirname(path)
            os.makedirs(dirname, exist_ok=True)
            img.save(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('url', type=str, help='data source url')
    parser.add_argument('-c', '--data_parse_config', default='data_parse.json')
    args = parser.parse_args()

    url = args.url
    data_parse_config_file = args.data_parse_config
    with open(data_parse_config_file, 'r') as f:
        data_parse_config = json.load(f)

    data_config = data_parse_config['html']

    data_from_url = DataFromUrl(url, encoding='utf8', filter_tags=data_config["filter_tags"])
    html_list, image_src = data_from_url.get_data(
        replace_tags=data_config["replace_tags"],
        decompose_tags=data_config["decompose_tags"],
        preserve_tags=data_config["preserve_tags"],
        preserve_attrs=data_config["preserve_attrs"],
        save_filename=data_config["output_file"],
    )
