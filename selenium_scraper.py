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

try:
    import urlparse
    from urllib import urlencode
except:  # For Python 3
    import urllib.parse as urlparse
    from urllib.parse import urlencode


def build_query(search_term, start_date, end_date=datetime.now().date()):
    """

    :param search_term: search LexisNexis for this
    :param start_date: start date (inclusive)
    :param end_date: end date (inclusive)
    :return: url
    """
    str = 'http://www.lexisnexis.com/lnacui2api/api/version1/sr?swn=t&icvrpg=true&oc=00006&hes=t&elb=t&hnsl=t&crth=off&hfb=t&hb=t&halltb=t&hgn=f&hdym=t&ssl=f&so=re&sr=%28bitcoin%29+and+Date%28geq%288%2F1%2F2017%29+and+leq%283%2F9%2F2018%29%29&stp=boolean&hsp=t&hso=t&hsl=t&hp=t&hcu=t&hh=t&hl=t&csi=8411&secondRedirectIndicator=true'
    parsed = urlparse.urlparse(str)
    params = urlparse.parse_qs(parsed.query)
    for key in params:
        if len(params[key]) == 1:
            params[key] = params[key][0]
    start_date_str = f'{start_date.month}/{start_date.day}/{start_date.year}'

    end_date_str = f'{end_date.month}/{end_date.day}/{end_date.year}'
    params['sr'] = f'({search_term}) and Date(geq({start_date_str}) and leq({end_date_str}))'  # m/d/year

    return 'http://www.lexisnexis.com/lnacui2api/api/version1/sr?' + urlencode(params)


def scrape(directory='results_html', query='microsoft', start_date=datetime.now().date(),
           end_date=datetime.now().date(), last_page=10000):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    additional_wait_time = 0.5
    # driver = webdriver.Chrome()
    opts = Options()
    opts.add_argument(f"user-agent={UserAgent().chrome}")
    opts.add_argument('headless')

    url = build_query(query, start_date, end_date)
    driver = webdriver.Chrome(chrome_options=opts)
    # driver.get('https://www.lexisnexis.com/hottopics/lnacademic/')
    driver.get(url)

    # input('Are you done navigating?')
    # driver.switch_to.frame('mainFrame')
    time.sleep(additional_wait_time)
    classification_frame_name = driver.find_element_by_xpath(
        '//frame[starts-with(@name, "fr_classification")]').get_attribute('name')
    driver.switch_to.frame(classification_frame_name)
    driver.find_element_by_xpath(
        '//a[starts-with(@title, "Newspapers ")]').click()
    driver.switch_to.parent_frame()
    time.sleep(additional_wait_time)

    for i in range(10000):
        if i == last_page:
            print(f"Reached last page: {last_page}")
            break
        results_contents_frame_name = driver.find_element_by_xpath(
            '//frame[starts-with(@name, "fr_resultsContent")]').get_attribute('name')
        results_navigator_frame_name = driver.find_element_by_xpath(
            '//frame[starts-with(@name, "fr_resultsNav")]').get_attribute('name')
        driver.switch_to.frame(results_contents_frame_name)
        # scrape page...
        with open(f'{directory}/{i}.html', 'wb') as f:
            f.write(driver.page_source.encode('utf8'))
        # results_table_elem = driver.find_element_by_xpath('//table[@role="main"]')
        # results_table_elem.find_elements_by_xpath('.//a[@target="_parent"]')

        driver.switch_to.parent_frame()
        driver.switch_to.frame(results_navigator_frame_name)
        try:
            elem = driver.find_element_by_xpath('//a[@tabindex="17"]')
            elem.click()
        except NoSuchElementException:
            print("Can't find next button. At end?")
            break
            # stop now...
        time.sleep(additional_wait_time)
        driver.switch_to.parent_frame()

    driver.close()


def download_results_html(query='bitcoin', start_date=date(year=2017, month=1, day=1),
                          date_offset=dateutil.relativedelta.relativedelta(months=1),
                          last_page=10000):
    hard_start_date = False
    end_date = datetime.now().date()
    # end_date = date(year=1999, month=3, day=9)

    cur_start_date = end_date
    while True:
        cur_end_date = cur_start_date - dateutil.relativedelta.relativedelta(days=1)
        if hard_start_date:
            cur_start_date = max(start_date, cur_start_date - date_offset)
        else:
            cur_start_date = cur_start_date - date_offset

        directory = f'output/{query}/{cur_start_date}_{cur_end_date}_html'
        scrape(directory, query=f'{query}', start_date=cur_start_date, end_date=cur_end_date,
               last_page=last_page)

        if cur_start_date <= start_date:
            break


def parse_results(query='bitcoin'):
    results = []

    for root, dirs, files in os.walk(os.path.join('output', query)):
        if len(files) == 0:
            continue
        page_numbers = np.array([int(x.replace('.html', '')) for x in files])
        last_page = float(page_numbers.max() * 0.4)
        for file in files:
            if int(file.replace('.html', '')) > last_page:
                continue
            tree = html.parse(os.path.join(root, file))
            titles = tree.xpath('//a[@target="_parent"]/text()')
            contents = tree.xpath('//a[@target="_parent"]/../text()')
            results += [(t, c) for t, c in zip(titles, contents)]

    months = "January, Jan, February, Feb, March, Mar, April, Apr, May, May, June, Jun, July, Jul, August, Aug, September, Sep, Sept, October, Oct, November, Nov, December, Dec"
    months = [x.strip() for x in months.split(',')]
    months = "|".join(months)
    print(months)
    date_re = re.compile(f'({months}) \\d+,? \\d+', re.RegexFlag.IGNORECASE)

    results_table = []
    for title, content in results:
        title = title.replace('"', "").replace('\t', ' ').strip()
        try:
            date = date_re.search(content).group()
        except AttributeError:
            print(f'No date found. title="{title}"; content="{content}"')
            continue
        date = parser.parse(date).strftime('%Y-%m-%d')
        results_table.append(dict(date=date, title=title))

    results_df = pd.DataFrame(results_table)
    results_df = results_df.groupby(['title']).min()
    results_df['title'] = results_df.index

    titles_per_day_df = results_df.groupby(['date']).apply(lambda x: x['title'].iloc[0]).sort_index()
    print(f'Distinct days downloaded = {len(titles_per_day_df)}')
    pd.Series.to_csv(titles_per_day_df, 'title_by_day.tsv', sep='\t')

    results_df.to_csv('all_titles_per_day.tsv', sep='\t', index=False)
    print(f'Number of titles downloaded = {len(results_df)}')
    # titles_per_day_df = results_df.groupby(['date']).apply(lambda x: '\n'.join(x['title'])).sort_index()


if __name__ == '__main__':
    query = 'nvidia'
    download_results_html(query, start_date=date(2018, month=1, day=1),
                          date_offset=dateutil.relativedelta.relativedelta(years=1),
                          last_page=30)
    parse_results(query)
