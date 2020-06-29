import requests
from bs4 import BeautifulSoup


url = 'https://fbref.com/en/matches/928467bd/Liverpool-Norwich-City-August-9-2019-Premier-League'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
team_stats = soup.find(id='team_stats')
stats = team_stats.find_all('strong')


def str_to_int(str):
    return int(str.replace('%', ''))


poss = (str_to_int(stats[0].text), str_to_int(stats[1].text))
pass_acc = (str_to_int(stats[2].text), str_to_int(stats[3].text))
print(poss, pass_acc)

team_stats_extra = soup.find(id='team_stats_extra')
stats_extra = team_stats_extra.find_all('div')
print(stats_extra)
