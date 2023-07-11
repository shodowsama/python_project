import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

database = '../data_analy/LV4_SQL_in_python/data.sqlite'

conn = sqlite3.connect(database)

# sqlite_master 為默認名稱
tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conn)
print(tables)
# 表格join
team = pd.read_sql('''select * from Team;''', conn)
TAri = pd.read_sql('''select * from Team_Attributes;''', conn)

team_api = pd.read_sql('''select Team_Attributes.defenceAggressionClass , Team_Attributes.defenceTeamWidth ,  Team_Attributes.date , Team.team_api_id , Team.team_fifa_api_id , Team.team_short_name
                                from Team_Attributes left join Team on Team_Attributes.team_api_id = team.team_api_id ;''', conn)

# player 排序
player = pd.read_sql('''select * from Player order by height desc;''', conn)

Match = pd.read_sql('''select * from Match;''', conn)
leag = pd.read_sql('''select * from League;''', conn)

"""
detail_player = pd.read_sql('''select Player.player_name, 
                                    Player.player_api_id as api, 
                                    Match.date, 
                                    Match.league_id, 
                                    Match.season,
                                    League.name
                                    from Match 
                                    join League  on Match.country_id = League.country_id
                                    where stage = 1
                                    order by date
                                    limit 10 ;
''', conn)
"""
# 計算League 差異
detail_player = pd.read_sql('''select  case 
                                    when round(height)<165 then 165
                                    when round(height)>195 then 195
                                    else round(height)
                                    end as calc_height,
                                    count(height),
                                    avg(PAG.avg_overall_rating),
                                    avg(weight)
                                from Player left join (
                                    select player_api_id,
                                    avg(overall_rating) as avg_overall_rating
                                    from Player_Attributes
                                    group by player_api_id) as PAG
                                on Player.player_api_id = PAG.player_api_id
                                group by calc_height
;''', conn)

detail_player.plot(x='calc_height', y='avg(PAG.avg_overall_rating)')
plt.show()
