import sys
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from datetime import date, datetime
import dateutil.relativedelta
import os, errno
from lxml import html
import re
import os
from dateutil import parser
import pandas as pd
import numpy as np
import math
import sqlite3
from contextlib import contextmanager
import uuid
from bs4 import BeautifulSoup
from pattern.web import plaintext
from joblib import Parallel, delayed

try:
    import urlparse
    from urllib import urlencode
except:  # For Python 3
    import urllib.parse as urlparse
    from urllib.parse import urlencode


class SeleniumScraper:
    MONTHS = "January, Jan, February, Feb, March, Mar, April, Apr, May, May, June, Jun, July, Jul, August, Aug, September, Sep, Sept, October, Oct, November, Nov, December, Dec"
    MONTHS = [x.strip() for x in MONTHS.split(',')]
    MONTHS = "|".join(MONTHS)
    DATE_RE = re.compile(f'({MONTHS}) \\d+,? \\d+', re.RegexFlag.IGNORECASE)
    NUMBER_RE = re.compile(r'((?<=\s)|^).?\d+.?((?=\s)|$)')
    URL_RE = re.compile(
        r'(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')
    ARTICLE_INFO_WORDS = ['BYLINE', 'SECTION', 'LENGTH', 'RELATED', 'LANGUAGE', 'HIGHLIGHT', 'GRAPHIC',
                          'PUBLICATION-TYPE']
    ARTICLE_INFO_LINE_RE = re.compile(f'({"|".join(ARTICLE_INFO_WORDS)}):')

    def __init__(self, search_term, start_date, end_date=datetime.now().date(),
                 date_offset=dateutil.relativedelta.relativedelta(months=1), last_page=10000, min_words=5,
                 scrape_articles=False):
        self.scrape_articles = scrape_articles
        self.search_term = search_term
        self.start_date = start_date
        self.end_date = end_date
        self.last_page = last_page
        self.date_offset = date_offset
        self.min_words = min_words
        self.driver = SeleniumScraper.create_webdriver()

    @staticmethod
    def create_webdriver():
        # driver = webdriver.Chrome()
        opts = Options()
        opts.add_argument(f"user-agent={UserAgent().chrome}")
        opts.add_argument('headless')

        return webdriver.Chrome(chrome_options=opts)

    def __del__(self):
        self.driver.close()
        with sqlite3.connect('articles.db') as conn:
            cur = conn.cursor()
            cur.execute(r"""select 'drop table `' || name || '`;' from sqlite_master
    where type = 'table'
	and name like "\__tmp_%" escape '\'""")
            for row in cur.fetchall():
                cur.execute(row[0])

    def query_lexis_nexis(self, start_date, end_date, driver=None):
        if driver is None:
            driver = SeleniumScraper.create_webdriver()
        additional_wait_time = 0.5

        self.set_up_query(self.search_term, start_date, end_date)
        # driver.get('https://www.lexisnexis.com/hottopics/lnacademic/')
        # driver.get(url)

        # input('Are you done navigating?')
        # driver.switch_to.frame('mainFrame')
        time.sleep(additional_wait_time)
        try:
            classification_frame_name = driver.find_element_by_xpath(
                '//frame[starts-with(@name, "fr_classification")]').get_attribute('name')
        except NoSuchElementException as e:
            raise Exception("Are you sure you are connected to a network with access to LexisNexis?").with_traceback(
                sys.exc_info()[2])

        driver.switch_to.frame(classification_frame_name)
        driver.find_element_by_xpath(
            '//a[starts-with(@title, "Newspapers ")]').click()
        driver.switch_to.parent_frame()
        time.sleep(additional_wait_time)
        results_navigator_frame_name = driver.find_element_by_xpath(
            '//frame[starts-with(@name, "fr_resultsNav")]').get_attribute('name')
        # driver.switch_to.frame(results_navigator_frame_name)
        # driver.find_element_by_xpath(
        #     '//select[@name="thresholdType"]/option[@value="search.common.threshold.broadrange"]').click()
        # time.sleep(additional_wait_time)
        # driver.switch_to.parent_frame()

        for i in range(10000):
            results_contents_frame_name = driver.find_element_by_xpath(
                '//frame[starts-with(@name, "fr_resultsContent")]').get_attribute('name')
            results_navigator_frame_name = driver.find_element_by_xpath(
                '//frame[starts-with(@name, "fr_resultsNav")]').get_attribute('name')
            driver.switch_to.frame(results_contents_frame_name)
            # scrape page...
            yield driver.page_source.encode('utf8')
            # results_table_elem = driver.find_element_by_xpath('//table[@role="main"]')
            # results_table_elem.find_elements_by_xpath('.//a[@target="_parent"]')

            driver.switch_to.parent_frame()
            driver.switch_to.frame(results_navigator_frame_name)
            if i + 1 == self.last_page:
                print(f"Reached last page: {self.last_page}")
                break
            try:
                elem = driver.find_element_by_xpath('//a[@tabindex="17"]')
                elem.click()
            except NoSuchElementException:
                print("Can't find next button. At end?")
                break
                # stop now...
            time.sleep(additional_wait_time)
            driver.switch_to.parent_frame()

    def set_up_query(self, search_term, start_date, end_date):
        """

        :param search_term: search LexisNexis for this
        :param start_date: start date (inclusive)
        :param end_date: end date (inclusive)
        :return: url
        """
        start_date_str = f'{start_date.year}-{start_date.month}-{start_date.day}'
        end_date_str = f'{end_date.year}-{end_date.month}-{end_date.day}'

        self.driver.get('http://www.lexisnexis.com/lnacui2api')
        self.driver.find_element_by_xpath(
            '//select[@name="thresholdType"]/option[@value="search.common.threshold.broadrange"]').click()
        self.driver.find_element_by_xpath(
            '//select[@name="dateSelector"]/option[@value="All"]').click()
        self.driver.find_element_by_xpath('//textarea[@name="searchTermsTextArea"]').send_keys(
            f"DATE(>={start_date_str} and <={end_date_str}) and ({self.search_term})")
        self.driver.execute_script("javascript:onSearch('imagesearch')")

        # str = 'http://www.lexisnexis.com/lnacui2api/api/version1/sr?swn=t&icvrpg=true&oc=00006&hes=t&elb=t&hnsl=t&crth=off&hfb=t&hb=t&halltb=t&hgn=f&hdym=t&ssl=f&so=re&sr=%28bitcoin%29+and+Date%28geq%288%2F1%2F2017%29+and+leq%283%2F9%2F2018%29%29&stp=boolean&hsp=t&hso=t&hsl=t&hp=t&hcu=t&hh=t&hl=t&csi=8411&secondRedirectIndicator=true'
        # parsed = urlparse.urlparse(str)
        # params = urlparse.parse_qs(parsed.query)
        # for key in params:
        #     if len(params[key]) == 1:
        #         params[key] = params[key][0]
        # start_date_str = f'{start_date.month}/{start_date.day}/{start_date.year}'
        #
        # end_date_str = f'{end_date.month}/{end_date.day}/{end_date.year}'
        # params['sr'] = f'({search_term}) and Date(geq({start_date_str}) and leq({end_date_str}))'  # m/d/year
        #
        # return 'http://www.lexisnexis.com/lnacui2api/api/version1/sr?' + urlencode(params)

    from joblib import Parallel, delayed

    def get_all_search_page_html(self):
        if self.last_page is None:
            last_page = 10000
        hard_start_date = False
        # end_date = date(year=1999, month=3, day=9)

        cur_start_date = self.end_date
        query_parameters = []
        while True:
            cur_end_date = cur_start_date - dateutil.relativedelta.relativedelta(days=1)
            if hard_start_date:
                cur_start_date = max(self.start_date, cur_start_date - self.date_offset)
            else:
                cur_start_date = cur_start_date - self.date_offset

            search_range = f'{cur_start_date} to {cur_end_date}'

            query_parameters.append((cur_start_date, cur_end_date, self.driver, search_range))
            # yield self.query_lexis_nexis(start_date=cur_start_date, end_date=cur_end_date), search_range

            if cur_start_date <= self.start_date:
                break

        # Parallel(n_jobs=1)(delayed(self.query_lexis_nexis)(*params) for params in query_parameters)
        for *params, search_range in query_parameters:
            yield self.query_lexis_nexis(*params), search_range

    @staticmethod
    def get_results_from_search_page(page_source):
        # tree = html.parse(page_source)
        tree = html.fromstring(page_source)
        # titles = tree.xpath('//a[@target="_parent"]/text()')
        titles = []
        date_strings = []
        contents = []
        article_urls = []
        for elem in tree.xpath('//a[@target="_parent"]/..'):
            a_elem = elem.xpath('./a')[0]
            content = html.tostring(elem, method='text', encoding='utf8')
            title = a_elem.xpath('./text()')
            if len(title) == 0:
                continue
            title = title[0]
            content_split = content.split(b'\xc2\xa0')
            if len(content_split) > 1:
                date_str = b' '.join(content_split[1:]).decode('utf8')
            else:
                date_str = ''

            contents.append(content.decode('utf8'))
            titles.append(title)
            date_strings.append(date_str)
            article_urls.append(urlparse.urljoin(a_elem.base, a_elem.attrib['href']))
        # contents = [html.tostring(x, method='text', encoding='utf8') for x in
        #             tree.xpath('//a[@target="_parent"]/..')]
        # titles = [x.split(b'\xc2\xa0')[0].decode('utf8') for x in contents]
        # contents = tree.xpath('//a[@target="_parent"]/../text()')
        # for x in contents:
        #     x_split = x.split(b'\xc2\xa0')
        #     if len(x_split) > 1:
        #         date_strings.append(b' '.join(x_split[1:]).decode('utf8'))
        #     else:
        #         date_strings.append('')
        # date_strings = [x.split(b'\xc2\xa0')[1].decode('utf8') for x in contents]
        # contents = [x.decode('utf8') for x in contents]
        # article_urls = [urlparse.urljoin(x.base, x.attrib['href']) for x in tree.xpath('//a[@target="_parent"]')]
        assert len(contents) == len(date_strings)
        assert len(date_strings) == len(titles)
        assert len(article_urls) == len(titles)
        return [(t, d, c, u) for t, d, c, u in zip(titles, date_strings, contents, article_urls)]

    @staticmethod
    def filter_text(text):
        text = text.replace('"', "").replace('\t', ' ').replace(',', '').replace('.', '').lower()
        text = SeleniumScraper.NUMBER_RE.sub('', text).strip()
        return text

    def raw_results_to_df(self, results):
        results_table = []
        for title, date_str, content, url in results:
            title = SeleniumScraper.filter_text(title)
            num_words = len(re.findall('\w+', title))
            if self.min_words is not None and num_words < self.min_words:
                continue
            try:
                date = SeleniumScraper.DATE_RE.search(date_str).group()
            except AttributeError:
                print(f'No date found. title="{title}"; date_str={date_str}; content="{content}"')
                continue
            try:  # TODO: better solution
                date = parser.parse(date).strftime('%Y-%m-%d')
            except ValueError:
                continue
            results_table.append(dict(date=date, title=title, url=url))
        results_df = pd.DataFrame(results_table)
        results_df = results_df.groupby(['title']).min()
        results_df['title'] = results_df.index
        results_df['query'] = self.search_term
        return results_df

    def scrape_results_to_db(self, verbose=False):
        for date_offset_iter, search_range in self.get_all_search_page_html():
            if verbose:
                print(search_range)
            with sqlite3.connect('articles.db') as conn:
                cur = conn.cursor()
                cur.execute(f"""select 1 from article
where query = '{self.search_term}'
and search_range = '{search_range}'
limit 1""")
                if len(cur.fetchall()) > 0:
                    if verbose:
                        print('\tskipping because date range exists in database...')
                    continue
            for page_num, search_page_source in enumerate(date_offset_iter):
                if verbose:
                    print('\t' + str(page_num))
                results = SeleniumScraper.get_results_from_search_page(search_page_source)
                if len(results) == 0:
                    continue
                results_df = self.raw_results_to_df(results)
                results_df['page_num'] = page_num
                results_df['search_range'] = search_range
                self.add_results_to_db(results_df)

    def parse_results(self, ratio_to_keep=0.8, page_limit=None, min_words=5):
        if page_limit is None:
            page_limit = math.inf
        results = []

        for root, dirs, files in os.walk(os.path.join('output', self.search_term)):
            if len(files) == 0:
                continue
            page_numbers = np.array([int(x.replace('.html', '')) for x in files])
            last_page = min(float(page_numbers.max() * ratio_to_keep), page_limit)
            for file in files:
                if int(file.replace('.html', '')) > last_page:
                    continue
                results += SeleniumScraper.get_results_from_search_page(file)

        results_table = []
        for title, date_str, content, url in results:
            title = title.replace('"', "").replace('\t', ' ').replace(',', '').replace('.', '').lower()
            title = SeleniumScraper.NUMBER_RE.sub('', title).strip()
            num_words = len(re.findall('\w+', title))
            if min_words is not None and num_words < min_words:
                continue
            try:
                date = SeleniumScraper.DATE_RE.search(date_str).group()
            except AttributeError:
                print(f'No date found. title="{title}"; date_str={date_str}; content="{content}"')
                continue
            try:  # TODO: better solution
                date = parser.parse(date).strftime('%Y-%m-%d')
            except ValueError:
                continue
            results_table.append(dict(date=date, title=title, url=url))

        results_df = pd.DataFrame(results_table)
        results_df = results_df.groupby(['title']).min()
        results_df['title'] = results_df.index

        titles_per_day_df = results_df.groupby(['date']).apply(lambda x: x['title'].iloc[0]).sort_index()
        print(f'Distinct days downloaded = {len(titles_per_day_df)}')
        pd.Series.to_csv(titles_per_day_df, 'title_by_day.tsv', sep='\t', encoding='utf8')

        results_df.to_csv('all_titles_per_day.tsv', sep='\t', index=False, encoding='utf8')
        print(f'Number of titles downloaded = {len(results_df)}')
        # titles_per_day_df = results_df.groupby(['date']).apply(lambda x: '\n'.join(x['title'])).sort_index()
        results_df['query'] = self.search_term
        return results_df

    @staticmethod
    @contextmanager
    def tmp_table(df, conn):
        name = '__tmp_' + str(uuid.uuid4())
        df.to_sql(name, conn, index=False)
        yield name
        cur = conn.cursor()
        cur.execute(f'drop table `{name}`')

    def add_results_to_db(self, results_df):
        def get_contents(url):
            print(url)
            self.driver.get(url)
            time.sleep(0.1)
            with self.switch_to_frame('fr_resultsContent'):
                return self.driver.page_source.encode('utf8')

        df = results_df[['query', 'title', 'date', 'url', 'search_range', 'page_num']]
        with sqlite3.connect('articles.db') as conn:
            with SeleniumScraper.tmp_table(df, conn) as tmp_name:
                cur = conn.cursor()
                cur.execute(f"""select t1.query, t1.title, t1.date, t1.url, t1.search_range, t1.page_num from `{tmp_name}` as t1 left join article as t2 on t1.title = t2.title and t1.query = t2.query
where t1.date < t2.date or t2.html is null""")
                if self.scrape_articles:
                    results_to_update = []
                    with self.new_tab():
                        for query, title, date, url, search_range, page_num in cur.fetchall():
                            article_source = get_contents(url)
                            results_to_update.append((query, title, date, article_source, None, search_range, page_num))
                    cur.executemany('replace into article values (?, ?, ?, ?, ?, ?, ?)', results_to_update)
                else:
                    cur.execute(f"""
        replace into article
select query, title, date, NULL, NULL, search_range, page_num from `{tmp_name}` as t1
where not exists (select 1 from article where t1.query = query and t1.title = title and t1.date = date)""")  # only add new values

    @contextmanager
    def switch_to_frame(self, frame_name):
        frame_name = self.driver.find_element_by_xpath(
            f'//frame[starts-with(@name, "{frame_name}")]').get_attribute('name')
        self.driver.switch_to.frame(frame_name)
        print(frame_name)
        yield
        self.driver.switch_to.parent_frame()
        print('parent frame...')

    @contextmanager
    def new_tab(self):
        prev_window = self.driver.window_handles[0]
        self.driver.execute_script("window.open('about:blank', 'tab2');")
        self.driver.switch_to.window("tab2")
        yield
        self.driver.close()
        self.driver.switch_to.window(prev_window)

    def download_missing_html(self):
        def get_contents(url):
            print(url)
            self.driver.get(url)
            time.sleep(0.1)
            with self.switch_to_frame('fr_resultsContent'):
                return self.driver.page_source.encode('utf8')

        with sqlite3.connect('articles.db') as conn:
            df = pd.read_sql(sql='select * from article where query="nvidia" and html is NULL', con=conn)
            print(df)
            if len(df) == 0:
                return

            # opts = Options()
            # opts.add_argument(f"user-agent={UserAgent().chrome}")
            # opts.add_argument('headless')

            # driver = webdriver.Chrome(chrome_options=opts)

            # with open('tmp.html', 'wb') as f:
            #     f.write(get_contents(url))
            cur = conn.cursor()
            with self.new_tab():
                for i, row in df.iterrows():
                    # df.iloc[i]['html'] = get_contents(row['url'])
                    html_content = get_contents(row['url'])
                    cur.execute(f'REPLACE INTO article (query, title, date, url, html) VALUES (?, ?, ?, ?, ?)',
                                (row['query'], row['title'], row['date'], row['url'], html_content))

    @staticmethod
    def write_html(bhtml, name='tmp'):
        with open(f'{name}.html', 'wb') as f:
            f.write(bhtml)

    @staticmethod
    def parse_article_html(query=None):
        with sqlite3.connect('articles.db') as conn:
            conn.row_factory = sqlite3.Row
            #             articles_df = pd.read_sql(sql=f"""select * from article as t1
            # where exists(select 1 from `{tmp_name}` as t2
            # where t1.query = t2.query and t1.title = t2.title and t1.date = t2.date)""", con=conn)
            cur = conn.cursor()
            update_cur = conn.cursor()
            cur.execute(f"""select query, title, date, html from article as t1 
    where {f'query = "{query}" and' if query is not None else ''} html is not null""")
            cur_page = 0
            batch_size = 100
            while True:
                print('done up to', cur_page)
                results = cur.fetchmany(100)
                cur_page += batch_size
                if not results:
                    break
                full_text_update_parameters = []
                for result in results:
                    # tree = html.fromstring(result['html'])
                    # content_elems = tree.xpath('//span[@class="verdana"]')
                    # assert len(content_elems) == 1
                    # print([url_re.sub(b' ', html.tostring(x, method='text', encoding='utf8')) for x in content_elems[0].xpath('./p')])
                    # print(BeautifulSoup(content_elems[0]))
                    # soup = BeautifulSoup(result['html'])
                    # write_html(old_html, 'tmp')
                    tree = html.fromstring(result['html'])
                    for bad in tree.xpath('//span[@id="crosslinktitlebar"]/../*'):
                        bad.getparent().remove(bad)
                    content_elems = tree.xpath('//span[@class="verdana"]')
                    assert len(content_elems) == 1
                    for bad in content_elems[0].xpath('.//table'):
                        bad.getparent().remove(bad)
                    # write_html(html.tostring(tree), 'tmp2')
                    html_str = html.tostring(content_elems[0])
                    article_text = plaintext(html_str.decode('utf8'), keep={}, linebreaks=1)
                    text_list = []
                    for line_index, line in enumerate(article_text.split('\n')):
                        if line_index == 0:  # title of the article
                            continue
                        line = line.strip()
                        if SeleniumScraper.DATE_RE.fullmatch(line):
                            continue
                        if SeleniumScraper.ARTICLE_INFO_LINE_RE.match(line):
                            continue
                        line = SeleniumScraper.DATE_RE.sub(' ', line)
                        line = SeleniumScraper.URL_RE.sub(' ', line)
                        line = re.sub(r'\[[A-Za-z=\- 0-9]+\]', ' ', line)
                        line = line.replace('\t', ' ')
                        line = line.strip()
                        if line == '':
                            continue
                        if 'ibd-display-video' in line:
                            SeleniumScraper.write_html(html_str)
                        text_list.append(line)
                    full_text_update_parameters.append(
                        (SeleniumScraper.filter_text(' '.join(text_list)), result['query'], result['title'], result['date']))
                    # print(html.tostring(tree.xpath('//span[@class="verdana"]')[0], method='text', encoding='utf8').decode('utf8'))
                # df = pd.DataFrame(data=full_text_update_commands, columns=['date', 'article'])
                # if b_add_header:
                #     df.to_csv('articles_per_day.tsv', sep='\t', encoding='utf8', index=False)
                #     b_add_header = False
                # else:
                #     df.to_csv('articles_per_day.tsv', sep='\t', mode='a', encoding='utf8', header=False,
                #               index=False)
                update_cur.executemany(
                    """update article set full_text = (?)
                    where query = (?)
                    and title = (?)
                    and date = (?)""", full_text_update_parameters)


if __name__ == '__main__':
    query = 'bitcoin'
    # download_results_html(query, start_date=date(2017, month=1, day=1),
    #                       date_offset=dateutil.relativedelta.relativedelta(years=2),
    #                       last_page=1)
    # results = parse_results(query, 0.6, page_limit=20, min_words=5)
    # add_results_to_db(query, results)

    SeleniumScraper.parse_article_html()

    # for i in range(100):
    #     try:
    #         scraper = SeleniumScraper(search_term='bitcoin', start_date=date(2016, month=1, day=1),
    #                                   date_offset=dateutil.relativedelta.relativedelta(months=1),
    #                                   last_page=50, scrape_articles=True, end_date=date(2018, 4, 6))
    #         scraper.scrape_results_to_db(verbose=True)
    #
    #         scraper = SeleniumScraper(search_term='microsoft', start_date=date(2000, month=1, day=1),
    #                                   date_offset=dateutil.relativedelta.relativedelta(months=6),
    #                                   last_page=50, scrape_articles=True, end_date=date(2018, 4, 6))
    #         scraper.scrape_results_to_db(verbose=True)
    #
    #         scraper = SeleniumScraper(search_term='trump', start_date=date(2016, month=1, day=1),
    #                                   date_offset=dateutil.relativedelta.relativedelta(months=1),
    #                                   last_page=50, scrape_articles=True, end_date=date(2018, 4, 6))
    #         scraper.scrape_results_to_db(verbose=True)
    #     except:
    #         if i >= 5:
    #             raise
    #     if i >= 5:
    #         break
