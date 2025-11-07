
# # """
# # FBref Data Scraper - Complete Version with Lineups & Events VERSION 1
# # """

# # import cloudscraper
# # from bs4 import BeautifulSoup
# # import pandas as pd
# # import time
# # import psycopg2
# # from datetime import datetime
# # import os
# # from dotenv import load_dotenv
# # from tqdm import tqdm
# # import random
# # import re

# # load_dotenv()

# # # Increased rate limits to avoid 403
# # RATE_LIMIT_MIN = 5
# # RATE_LIMIT_MAX = 8

# # class FBrefScraper:
# #     def __init__(self):
# #         self.scraper = cloudscraper.create_scraper(
# #             browser={
# #                 'browser': 'chrome',
# #                 'platform': 'windows',
# #                 'mobile': False
# #             },
# #             delay=10
# #         )
# #         self.base_url = 'https://fbref.com'
# #         self.conn = None
        
# #     def connect_db(self):
# #         """Connect to PostgreSQL database"""
# #         try:
# #             self.conn = psycopg2.connect(
# #                 host=os.getenv('POSTGRES_HOST', 'localhost'),
# #                 port=os.getenv('POSTGRES_PORT', '5432'),
# #                 database=os.getenv('POSTGRES_DB', 'spark_db'),
# #                 user=os.getenv('POSTGRES_USER', 'spark_user'),
# #                 password=os.getenv('POSTGRES_PASSWORD', 'spark_password_2024'),
# #                 connect_timeout=10
# #             )
# #             print("✅ Connected to database")
# #         except Exception as e:
# #             print(f"❌ Database connection failed: {e}")
# #             raise
        
# #     def close_db(self):
# #         if self.conn:
# #             self.conn.close()
# #             print("✅ Database connection closed")
    
# #     def truncate_all_tables(self):
# #         """Truncate all tables before fresh scrape"""
# #         cur = self.conn.cursor()
# #         try:
# #             print("\n🗑️  Truncating all tables...")
# #             tables = [
# #                 'MATCH_PREDICTIONS', 'MATCH_EVENT', 'MATCH_LINEUPS', 'MATCHES',
# #                 'TEAM_SEASONS', 'PLAYER_NICKNAMES', 'STAFF', 'PLAYERS', 
# #                 'TEAMS', 'SEASONS', 'LEAGUES'
# #             ]
# #             for table in tables:
# #                 cur.execute(f"TRUNCATE TABLE {table} CASCADE")
# #             self.conn.commit()
# #             print("✅ All tables truncated")
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error truncating tables: {e}")
# #         finally:
# #             cur.close()
    
# #     def random_delay(self):
# #         delay = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
# #         time.sleep(delay)
    
# #     def extract_fbref_id(self, url):
# #         """Extract FBref ID from URL"""
# #         match = re.search(r'/([a-f0-9]{8})/', url)
# #         if match:
# #             return match.group(1)
# #         return None
    
# #     def scrape_league_standings(self, league_url, season_year):
# #         """Scrape league standings with team URLs and FBref IDs"""
# #         print(f"\n📊 Scraping standings from: {league_url}")
# #         self.random_delay()
        
# #         try:
# #             response = self.scraper.get(league_url)
# #             response.raise_for_status()
# #             soup = BeautifulSoup(response.content, 'html.parser')
            
# #             table = soup.find('table', {'id': re.compile(r'.*standings.*', re.I)})
# #             if not table:
# #                 tables = soup.find_all('table')
# #                 for t in tables:
# #                     if 'Squad' in str(t):
# #                         table = t
# #                         break
            
# #             if not table:
# #                 print("❌ Could not find standings table")
# #                 return pd.DataFrame()
            
# #             df = pd.read_html(str(table))[0]
            
# #             if isinstance(df.columns, pd.MultiIndex):
# #                 df.columns = df.columns.droplevel(0)
            
# #             squad_col = None
# #             for col in df.columns:
# #                 if 'Squad' in str(col) or 'Team' in str(col):
# #                     squad_col = col
# #                     break
            
# #             if not squad_col:
# #                 return pd.DataFrame()
            
# #             df = df[df[squad_col] != squad_col]
# #             df = df.rename(columns={squad_col: 'Squad'})
            
# #             team_links = table.find_all('a', href=re.compile(r'/en/squads/'))
            
# #             team_data = []
# #             for link in team_links:
# #                 team_name = link.text.strip()
# #                 team_url = self.base_url + link['href']
# #                 fbref_id = self.extract_fbref_id(link['href'])
# #                 team_data.append({
# #                     'Squad': team_name,
# #                     'team_url': team_url,
# #                     'fbref_id': fbref_id if fbref_id else team_url
# #                 })
            
# #             team_df = pd.DataFrame(team_data)
# #             df = df.merge(team_df, on='Squad', how='left')
            
# #             print(f"✅ Found {len(df)} teams with URLs")
# #             return df
            
# #         except Exception as e:
# #             print(f"❌ Error scraping standings: {e}")
# #             return pd.DataFrame()
    
# #     def scrape_team_players(self, team_url):
# #         """Scrape player data for a specific team"""
# #         print(f"\n👥 Scraping players from: {team_url}")
# #         self.random_delay()
        
# #         try:
# #             response = self.scraper.get(team_url)
# #             response.raise_for_status()
# #             soup = BeautifulSoup(response.content, 'html.parser')
            
# #             table = soup.find('table', {'id': re.compile(r'.*stats_standard.*', re.I)})
# #             if not table:
# #                 return pd.DataFrame()
            
# #             df = pd.read_html(str(table))[0]
            
# #             if isinstance(df.columns, pd.MultiIndex):
# #                 df.columns = ['_'.join(str(col)).strip() for col in df.columns.values]
            
# #             df = df[df.iloc[:, 0] != 'Rk']
            
# #             player_links = table.find_all('a', href=re.compile(r'/en/players/'))
            
# #             player_data = []
# #             for link in player_links:
# #                 player_name = link.text.strip()
# #                 player_url = self.base_url + link['href']
# #                 fbref_id = self.extract_fbref_id(link['href'])
# #                 player_data.append({
# #                     'player_name': player_name,
# #                     'player_url': player_url,
# #                     'fbref_id': fbref_id if fbref_id else player_url
# #                 })
            
# #             player_df = pd.DataFrame(player_data)
            
# #             name_col = None
# #             for col in df.columns:
# #                 if 'Player' in col:
# #                     name_col = col
# #                     break
            
# #             if name_col and not player_df.empty:
# #                 df = df.merge(player_df, left_on=name_col, right_on='player_name', how='left')
            
# #             print(f"✅ Found {len(df)} players")
# #             return df
# #         except Exception as e:
# #             print(f"❌ Error scraping players: {e}")
# #             return pd.DataFrame()
    
# #     def scrape_match_results(self, league_url, season_year):
# #         """Scrape match results with scores, xG, and match URLs"""
# #         fixtures_url = league_url.replace('Premier-League-Stats', 'schedule/Premier-League-Scores-and-Fixtures')
# #         print(f"\n⚽ Scraping matches from: {fixtures_url}")
# #         self.random_delay()
        
# #         try:
# #             response = self.scraper.get(fixtures_url)
# #             response.raise_for_status()
# #             soup = BeautifulSoup(response.content, 'html.parser')
            
# #             table = soup.find('table', {'id': re.compile(r'.*sched.*', re.I)})
# #             if not table:
# #                 tables = soup.find_all('table')
# #                 if tables:
# #                     table = tables[0]
            
# #             if not table:
# #                 return pd.DataFrame()
            
# #             df = pd.read_html(str(table))[0]
            
# #             if isinstance(df.columns, pd.MultiIndex):
# #                 df.columns = df.columns.droplevel(0)
            
# #             if 'Score' in df.columns:
# #                 df = df[df['Score'].notna()]
# #                 df = df[df['Score'] != 'Score']
            
# #             match_links = table.find_all('a', href=re.compile(r'/en/matches/'))
            
# #             match_data = []
# #             for link in match_links:
# #                 match_url = self.base_url + link['href']
# #                 fbref_id = self.extract_fbref_id(link['href'])
# #                 match_data.append({
# #                     'match_url': match_url,
# #                     'fbref_id': fbref_id if fbref_id else match_url
# #                 })
            
# #             if match_data:
# #                 for i, row in enumerate(match_data[:len(df)]):
# #                     if i < len(df):
# #                         df.loc[df.index[i], 'match_url'] = row['match_url']
# #                         df.loc[df.index[i], 'fbref_id'] = row['fbref_id']
            
# #             print(f"✅ Found {len(df)} completed matches")
# #             return df
# #         except Exception as e:
# #             print(f"❌ Error scraping matches: {e}")
# #             return pd.DataFrame()
    
# #     def scrape_match_lineups_and_events(self, match_url, match_id):
# #         """Scrape lineups and events from a match page"""
# #         print(f"\n📋 Scraping lineups/events: {match_url}")
# #         self.random_delay()
        
# #         try:
# #             response = self.scraper.get(match_url)
# #             response.raise_for_status()
# #             soup = BeautifulSoup(response.content, 'html.parser')
            
# #             lineups = []
# #             events = []
            
# #             # Find lineup tables (home and away)
# #             lineup_divs = soup.find_all('div', class_=re.compile(r'lineup', re.I))
            
# #             for div in lineup_divs:
# #                 # Get team name
# #                 team_header = div.find_previous('h2')
# #                 team_name = team_header.text.strip() if team_header else None
                
# #                 if not team_name:
# #                     continue
                
# #                 # Get team_id
# #                 cur = self.conn.cursor()
# #                 cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (team_name,))
# #                 team_result = cur.fetchone()
# #                 cur.close()
                
# #                 if not team_result:
# #                     continue
                
# #                 team_id = team_result[0]
                
# #                 # Find all player links in lineup
# #                 player_links = div.find_all('a', href=re.compile(r'/en/players/'))
                
# #                 for idx, link in enumerate(player_links):
# #                     player_fbref_id = self.extract_fbref_id(link['href'])
                    
# #                     # Get player_id
# #                     cur = self.conn.cursor()
# #                     cur.execute("SELECT player_id FROM PLAYERS WHERE fbref_id = %s", (player_fbref_id,))
# #                     player_result = cur.fetchone()
# #                     cur.close()
                    
# #                     if player_result:
# #                         lineups.append({
# #                             'match_id': match_id,
# #                             'player_id': player_result[0],
# #                             'team_id': team_id,
# #                             'is_starter': idx < 11,  # First 11 are starters
# #                             'minutes_played': 90 if idx < 11 else 0  # Default
# #                         })
            
# #             # Scrape events (goals, cards, subs)
# #             event_tables = soup.find_all('table', {'id': re.compile(r'.*events.*', re.I)})
            
# #             for table in event_tables:
# #                 rows = table.find_all('tr')
                
# #                 for row in rows:
# #                     cells = row.find_all(['td', 'th'])
# #                     if len(cells) < 3:
# #                         continue
                    
# #                     # Extract minute, event type, player
# #                     minute_cell = cells[0].text.strip()
# #                     event_cell = cells[1].text.strip()
                    
# #                     # Parse minute
# #                     minute_match = re.search(r'(\d+)', minute_cell)
# #                     if not minute_match:
# #                         continue
# #                     minute = int(minute_match.group(1))
                    
# #                     # Find player link
# #                     player_link = row.find('a', href=re.compile(r'/en/players/'))
# #                     if not player_link:
# #                         continue
                    
# #                     player_fbref_id = self.extract_fbref_id(player_link['href'])
                    
# #                     # Get player_id and team_id
# #                     cur = self.conn.cursor()
# #                     cur.execute("""
# #                         SELECT p.player_id, p.team_id 
# #                         FROM PLAYERS p 
# #                         WHERE p.fbref_id = %s
# #                     """, (player_fbref_id,))
# #                     player_result = cur.fetchone()
# #                     cur.close()
                    
# #                     if player_result:
# #                         events.append({
# #                             'match_id': match_id,
# #                             'minute': minute,
# #                             'event_type': event_cell,
# #                             'player_id': player_result[0],
# #                             'team_id': player_result[1]
# #                         })
            
# #             return lineups, events
            
# #         except Exception as e:
# #             print(f"❌ Error scraping match details: {e}")
# #             return [], []
    
# #     def insert_league(self, name, country, tier, fbref_id):
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("""
# #                 INSERT INTO LEAGUES (name, country, tier, fbref_id)
# #                 VALUES (%s, %s, %s, %s)
# #                 ON CONFLICT (fbref_id) DO UPDATE 
# #                 SET name = EXCLUDED.name
# #                 RETURNING league_id
# #             """, (name, country, tier, fbref_id))
# #             league_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             print(f"✅ Inserted league: {name}")
# #             return league_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting league: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_season(self, year, start_date, end_date, league_id):
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("""
# #                 SELECT season_id FROM SEASONS 
# #                 WHERE year = %s AND league_id = %s
# #             """, (year, league_id))
# #             existing = cur.fetchone()
            
# #             if existing:
# #                 season_id = existing[0]
# #                 print(f"ℹ️  Season {year} already exists")
# #             else:
# #                 cur.execute("""
# #                     INSERT INTO SEASONS (year, start_date, end_date, league_id)
# #                     VALUES (%s, %s, %s, %s)
# #                     RETURNING season_id
# #                 """, (year, start_date, end_date, league_id))
# #                 season_id = cur.fetchone()[0]
# #                 print(f"✅ Inserted season: {year}")
            
# #             self.conn.commit()
# #             return season_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting season: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_team(self, name, stadium, city, fbref_id):
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("""
# #                 INSERT INTO TEAMS (name, stadium_name, city, fbref_id)
# #                 VALUES (%s, %s, %s, %s)
# #                 ON CONFLICT (fbref_id) DO UPDATE 
# #                 SET name = EXCLUDED.name
# #                 RETURNING team_id
# #             """, (name, stadium, city, fbref_id))
# #             team_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return team_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting team {name}: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_player(self, full_name, nationality, position, team_id, fbref_id, age=None, shirt_number=None):
# #         cur = self.conn.cursor()
# #         print(f"Attempting to insert: {full_name} | Position: {position} | Team: {team_id} | FBref: {fbref_id}")
# #         try:
# #             pos_map = {
# #                 'GK': 'Goalkeeper', 
# #                 'DF': 'Defender',
# #                 'MF': 'Midfielder', 
# #                 'FW': 'Forward'
# #             }
            
# #             if position and len(str(position)) >= 2:
# #                 pos_abbr = str(position)[:2].upper()
# #                 position = pos_map.get(pos_abbr, 'Midfielder')
# #             else:
# #                 position = 'Midfielder'
            
# #             dob = None
# #             if age and str(age).strip() and str(age) != 'nan':
# #                 try:
# #                     current_year = datetime.now().year
# #                     birth_year = current_year - int(float(age))
# #                     dob = f"{birth_year}-01-01"
# #                 except:
# #                     pass
            
# #             cur.execute("""
# #                 INSERT INTO PLAYERS (full_name, nationality, position, team_id, fbref_id, dob, shirt_number)
# #                 VALUES (%s, %s, %s, %s, %s, %s, %s)
# #                 ON CONFLICT (fbref_id) DO UPDATE 
# #                 SET team_id = EXCLUDED.team_id, 
# #                     shirt_number = EXCLUDED.shirt_number,
# #                     full_name = EXCLUDED.full_name
# #                 RETURNING player_id
# #             """, (full_name, nationality, position, team_id, fbref_id, dob, shirt_number))
# #             player_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return player_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting player {full_name}: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_match(self, match_date, venue, home_team, away_team, 
# #                     home_score, away_score, home_xg, away_xg, season_id, fbref_id):
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (home_team,))
# #             home_id = cur.fetchone()
# #             cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (away_team,))
# #             away_id = cur.fetchone()
            
# #             if not home_id or not away_id:
# #                 return None
            
# #             cur.execute("""
# #                 INSERT INTO MATCHES 
# #                 (match_date, venue, home_score_final, away_score_final, 
# #                  home_xg, away_xg, season_id, home_team_id, away_team_id, fbref_id)
# #                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
# #                 ON CONFLICT (fbref_id) DO UPDATE
# #                 SET home_score_final = EXCLUDED.home_score_final,
# #                     away_score_final = EXCLUDED.away_score_final
# #                 RETURNING match_id
# #             """, (match_date, venue, home_score, away_score, 
# #                   home_xg, away_xg, season_id, home_id[0], away_id[0], fbref_id))
# #             match_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return match_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting match: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_team_season(self, team_id, season_id, standings_row):
# #         cur = self.conn.cursor()
# #         try:
# #             points = int(standings_row.get('Pts', 0)) if 'Pts' in standings_row else 0
# #             wins = int(standings_row.get('W', 0)) if 'W' in standings_row else 0
# #             draws = int(standings_row.get('D', 0)) if 'D' in standings_row else 0
# #             losses = int(standings_row.get('L', 0)) if 'L' in standings_row else 0
# #             goals_for = int(standings_row.get('GF', 0)) if 'GF' in standings_row else 0
# #             goals_against = int(standings_row.get('GA', 0)) if 'GA' in standings_row else 0
# #             goal_diff = int(standings_row.get('GD', 0)) if 'GD' in standings_row else 0
            
# #             position = None
# #             for col in ['Rk', 'Pos', 'Position']:
# #                 if col in standings_row:
# #                     try:
# #                         position = int(standings_row[col])
# #                         break
# #                     except:
# #                         pass
            
# #             cur.execute("""
# #                 INSERT INTO TEAM_SEASONS 
# #                 (team_id, season_id, points, wins, draws, losses, 
# #                  goals_for, goals_against, goal_diff, position)
# #                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
# #                 ON CONFLICT (team_id, season_id) DO UPDATE
# #                 SET points = EXCLUDED.points, wins = EXCLUDED.wins
# #                 RETURNING team_season_id
# #             """, (team_id, season_id, points, wins, draws, losses,
# #                   goals_for, goals_against, goal_diff, position))
            
# #             team_season_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return team_season_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting team_season: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_lineup(self, lineup_data):
# #         """Insert lineup record"""
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("""
# #                 INSERT INTO MATCH_LINEUPS 
# #                 (match_id, player_id, team_id, is_starter, minutes_played)
# #                 VALUES (%s, %s, %s, %s, %s)
# #                 RETURNING lineup_id
# #             """, (
# #                 lineup_data['match_id'],
# #                 lineup_data['player_id'],
# #                 lineup_data['team_id'],
# #                 lineup_data['is_starter'],
# #                 lineup_data['minutes_played']
# #             ))
# #             lineup_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return lineup_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_event(self, event_data):
# #         """Insert match event"""
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("""
# #                 INSERT INTO MATCH_EVENT 
# #                 (match_id, minute, event_type, player_id, team_id)
# #                 VALUES (%s, %s, %s, %s, %s)
# #                 RETURNING event_id
# #             """, (
# #                 event_data['match_id'],
# #                 event_data['minute'],
# #                 event_data['event_type'],
# #                 event_data['player_id'],
# #                 event_data['team_id']
# #             ))
# #             event_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return event_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             return None
# #         finally:
# #             cur.close()


# # def main():
# #     scraper = FBrefScraper()
    
# #     try:
# #         scraper.connect_db()
# #     except:
# #         print("❌ Could not connect to database")
# #         return
    
# #     try:
# #         print("\n" + "="*50)
# #         print("TRUNCATING TABLES")
# #         print("="*50)
# #         scraper.truncate_all_tables()
        
# #         print("\n" + "="*50)
# #         print("INSERTING LEAGUE")
# #         print("="*50)
# #         league_id = scraper.insert_league('Premier League', 'England', 1, '9')
        
# #         if not league_id:
# #             return
        
# #         print("\n" + "="*50)
# #         print("INSERTING SEASON")
# #         print("="*50)
# #         season_id = scraper.insert_season('2024-25', '2024-08-16', '2025-05-25', league_id)
        
# #         if not season_id:
# #             return
        
# #         print("\n" + "="*50)
# #         print("SCRAPING TEAMS")
# #         print("="*50)
# #         league_url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
# #         standings_df = scraper.scrape_league_standings(league_url, '2024-25')
        
# #         if standings_df.empty:
# #             return
        
# #         team_ids = {}
# #         for _, row in tqdm(standings_df.iterrows(), total=len(standings_df), desc="Inserting teams"):
# #             team_name = row['Squad']
# #             fbref_id = row.get('fbref_id', f"team_{team_name.replace(' ', '_').lower()}")
            
# #             team_id = scraper.insert_team(team_name, None, None, fbref_id)
# #             if team_id:
# #                 team_ids[team_name] = {
# #                     'team_id': team_id,
# #                     'team_url': row.get('team_url'),
# #                     'standings_row': row
# #                 }
        
# #         print(f"\n✅ Inserted {len(team_ids)} teams")
        
# #         print("\n" + "="*50)
# #         print("INSERTING TEAM SEASON STATS")
# #         print("="*50)
# #         for team_name, team_info in tqdm(team_ids.items(), desc="Team seasons"):
# #             scraper.insert_team_season(team_info['team_id'], season_id, team_info['standings_row'])
        
# #         print("\n" + "="*50)
# #         print("SCRAPING PLAYERS (ALL TEAMS)")
# #         print("="*50)
# #         player_count = 0
# #         for team_name, team_info in tqdm(list(team_ids.items()), desc="Scraping players"):
# #             if not team_info.get('team_url'):
# #                 continue
            
# #             players_df = scraper.scrape_team_players(team_info['team_url'])
            
# #             if players_df.empty:
# #                 continue
            
# #             name_col = next((col for col in players_df.columns if 'Player' in col), None)
# #             age_col = next((col for col in players_df.columns if 'Age' in col), None)
# #             pos_col = next((col for col in players_df.columns if 'Pos' in col), None)
# #             nation_col = next((col for col in players_df.columns if 'Nation' in col), None)
# #             number_col = next((col for col in players_df.columns if '#' in col or 'Num' in col), None)
            
# #             for _, player in players_df.iterrows():
# #                 if name_col and name_col in player:
# #                     player_name = player[name_col]
# #                     age = player.get(age_col) if age_col else None
# #                     position = player.get(pos_col) if pos_col else 'MF'
# #                     nationality = player.get(nation_col) if nation_col else None
# #                     shirt_number = player.get(number_col) if number_col else None
# #                     fbref_id = player.get('fbref_id', f"player_{player_name.replace(' ', '_').lower()}")
# #                     print(f"Attempting insert for player: {player_name}")
# #                     player_id = scraper.insert_player(
# #                         player_name, nationality, position, 
# #                         team_info['team_id'], fbref_id, age, shirt_number
# #                     )
# #                     if player_id:
# #                         player_count += 1
        
# #         print(f"\n✅ Inserted {player_count} players")
        
# #         print("\n" + "="*50)
# #         print("SCRAPING MATCHES")
# #         print("="*50)
# #         matches_df = scraper.scrape_match_results(league_url, '2024-25')
        
# #         if matches_df.empty:
# #             print("⚠️  No matches found")
# #         else:
# #             match_count = 0
# #             match_ids_list = []
            
# #             for _, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Inserting matches"):
# #                 try:
# #                     score = str(row['Score']).split('–')
# #                     if len(score) != 2:
# #                         score = str(row['Score']).split('-')
                    
# #                     home_score = int(score[0].strip()) if len(score) == 2 else None
# #                     away_score = int(score[1].strip()) if len(score) == 2 else None
                    
# #                     home_xg = float(row.get('xG', 0)) if 'xG' in row and pd.notna(row.get('xG')) else None
# #                     away_xg = float(row.get('xG.1', 0)) if 'xG.1' in row and pd.notna(row.get('xG.1')) else None
                    
# #                     fbref_id = row.get('fbref_id', f"match_{row['Date']}_{row['Home']}_{row['Away']}")
                    
# #                     match_id = scraper.insert_match(
# #                         row['Date'], row.get('Venue', ''),
# #                         row['Home'], row['Away'],
# #                         home_score, away_score,
# #                         home_xg, away_xg,
# #                         season_id, fbref_id
# #                     )
                    
# #                     if match_id:
# #                         match_count += 1
# #                         match_ids_list.append({
# #                             'match_id': match_id,
# #                             'match_url': row.get('match_url')
# #                         })
# #                 except Exception as e:
# #                     print(f"⚠️  Skipped match: {e}")
# #                     continue
            
# #             print(f"\n✅ Inserted {match_count} matches")
            
# #             # NEW: Scrape lineups and events for first 10 matches
# #             print("\n" + "="*50)
# #             print("SCRAPING LINEUPS & EVENTS (First 10 matches)")
# #             print("="*50)
            
# #             lineup_count = 0
# #             event_count = 0
            
# #             for match_info in tqdm(match_ids_list[:10], desc="Match details"):
# #                 if not match_info.get('match_url'):
# #                     continue
                
# #                 lineups, events = scraper.scrape_match_lineups_and_events(
# #                     match_info['match_url'],
# #                     match_info['match_id']
# #                 )
                
# #                 # Insert lineups
# #                 for lineup in lineups:
# #                     if scraper.insert_lineup(lineup):
# #                         lineup_count += 1
                
# #                 # Insert events
# #                 for event in events:
# #                     if scraper.insert_event(event):
# #                         event_count += 1
            
# #             print(f"\n✅ Inserted {lineup_count} lineup entries")
# #             print(f"✅ Inserted {event_count} match events")
        
# #         print("\n" + "="*50)
# #         print("✅ SCRAPING COMPLETED!")
# #         print("="*50)
        
# #     except Exception as e:
# #         print(f"\n❌ Fatal error: {e}")
# #         import traceback
# #         traceback.print_exc()
# #     finally:
# #         scraper.close_db()


# # if __name__ == '__main__':
# #     main()

# # """
# # FBref Data Scraper - FIXED VERSION VERSION 2
# # """

# # import cloudscraper
# # from bs4 import BeautifulSoup
# # import pandas as pd
# # import time
# # import psycopg2
# # from datetime import datetime
# # import os
# # from dotenv import load_dotenv
# # from tqdm import tqdm
# # import random
# # import re

# # load_dotenv()

# # # Increased rate limits to avoid 403
# # RATE_LIMIT_MIN = 5
# # RATE_LIMIT_MAX = 8

# # class FBrefScraper:
# #     def __init__(self):
# #         self.scraper = cloudscraper.create_scraper(
# #             browser={
# #                 'browser': 'chrome',
# #                 'platform': 's',
# #                 'mobile': False
# #             },
# #             delay=10
# #         )
# #         self.base_url = 'https://fbref.com'
# #         self.conn = None
        
# #     def connect_db(self):
# #         """Connect to PostgreSQL database"""
# #         try:
# #             self.conn = psycopg2.connect(
# #                 host=os.getenv('POSTGRES_HOST', 'localhost'),
# #                 port=os.getenv('POSTGRES_PORT', '5432'),
# #                 database=os.getenv('POSTGRES_DB', 'spark_db'),
# #                 user=os.getenv('POSTGRES_USER', 'spark_user'),
# #                 password=os.getenv('POSTGRES_PASSWORD', 'spark_password_2024'),
# #                 connect_timeout=10
# #             )
# #             print("✅ Connected to database")
# #         except Exception as e:
# #             print(f"❌ Database connection failed: {e}")
# #             raise
        
# #     def close_db(self):
# #         if self.conn:
# #             self.conn.close()
# #             print("✅ Database connection closed")
    
# #     def truncate_all_tables(self):
# #         """Truncate all tables before fresh scrape"""
# #         cur = self.conn.cursor()
# #         try:
# #             print("\n🗑️  Truncating all tables...")
# #             tables = [
# #                 'MATCH_PREDICTIONS', 'MATCH_EVENT', 'MATCH_LINEUPS', 'MATCHES',
# #                 'TEAM_SEASONS', 'PLAYER_NICKNAMES', 'STAFF', 'PLAYERS', 
# #                 'TEAMS', 'SEASONS', 'LEAGUES'
# #             ]
# #             for table in tables:
# #                 cur.execute(f"TRUNCATE TABLE {table} CASCADE")
# #             self.conn.commit()
# #             print("✅ All tables truncated")
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error truncating tables: {e}")
# #         finally:
# #             cur.close()
    
# #     def random_delay(self):
# #         delay = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
# #         time.sleep(delay)
    
# #     def extract_fbref_id(self, url):
# #         """Extract FBref ID from URL"""
# #         match = re.search(r'/([a-f0-9]{8})/', url)
# #         if match:
# #             return match.group(1)
# #         return None
    
# #     def scrape_league_standings(self, league_url, season_year):
# #         """Scrape league standings with team URLs and FBref IDs"""
# #         print(f"\n📊 Scraping standings from: {league_url}")
# #         self.random_delay()
        
# #         try:
# #             response = self.scraper.get(league_url)
# #             response.raise_for_status()
# #             soup = BeautifulSoup(response.content, 'html.parser')
            
# #             table = soup.find('table', {'id': re.compile(r'.*standings.*', re.I)})
# #             if not table:
# #                 tables = soup.find_all('table')
# #                 for t in tables:
# #                     if 'Squad' in str(t):
# #                         table = t
# #                         break
            
# #             if not table:
# #                 print("❌ Could not find standings table")
# #                 return pd.DataFrame()
            
# #             df = pd.read_html(str(table))[0]
            
# #             if isinstance(df.columns, pd.MultiIndex):
# #                 df.columns = df.columns.droplevel(0)
            
# #             squad_col = None
# #             for col in df.columns:
# #                 if 'Squad' in str(col) or 'Team' in str(col):
# #                     squad_col = col
# #                     break
            
# #             if not squad_col:
# #                 return pd.DataFrame()
            
# #             df = df[df[squad_col] != squad_col]
# #             df = df.rename(columns={squad_col: 'Squad'})
            
# #             team_links = table.find_all('a', href=re.compile(r'/en/squads/'))
            
# #             team_data = []
# #             for link in team_links:
# #                 team_name = link.text.strip()
# #                 team_url = self.base_url + link['href']
# #                 fbref_id = self.extract_fbref_id(link['href'])
# #                 team_data.append({
# #                     'Squad': team_name,
# #                     'team_url': team_url,
# #                     'fbref_id': fbref_id if fbref_id else team_url
# #                 })
            
# #             team_df = pd.DataFrame(team_data)
# #             df = df.merge(team_df, on='Squad', how='left')
            
# #             print(f"✅ Found {len(df)} teams with URLs")
# #             return df
            
# #         except Exception as e:
# #             print(f"❌ Error scraping standings: {e}")
# #             return pd.DataFrame()
    
# #     def scrape_team_players(self, team_url):
# #         """Scrape player data for a specific team - FIXED"""
# #         print(f"\n👥 Scraping players from: {team_url}")
# #         self.random_delay()
        
# #         try:
# #             response = self.scraper.get(team_url)
# #             response.raise_for_status()
# #             soup = BeautifulSoup(response.content, 'html.parser')
            
# #             table = soup.find('table', {'id': re.compile(r'.*stats_standard.*', re.I)})
# #             if not table:
# #                 print("❌ Could not find player stats table")
# #                 return pd.DataFrame()
            
# #             df = pd.read_html(str(table))[0]
            
# #             # FIX: Flatten multi-level columns properly
# #             if isinstance(df.columns, pd.MultiIndex):
# #                 df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
            
# #             # Remove header rows that repeat
# #             df = df[df['Player'] != 'Player']
            
# #             # Get player links
# #             player_links = table.find_all('a', href=re.compile(r'/en/players/'))
            
# #             player_data = []
# #             for link in player_links:
# #                 player_name = link.text.strip()
# #                 player_url = self.base_url + link['href']
# #                 fbref_id = self.extract_fbref_id(link['href'])
# #                 if fbref_id:  # Only add if we have valid fbref_id
# #                     player_data.append({
# #                         'Player': player_name,
# #                         'player_url': player_url,
# #                         'fbref_id': fbref_id
# #                     })
            
# #             player_df = pd.DataFrame(player_data)
            
# #             # Merge on Player name
# #             if not player_df.empty and 'Player' in df.columns:
# #                 df = df.merge(player_df, on='Player', how='left')
# #                 # Remove rows without fbref_id
# #                 df = df[df['fbref_id'].notna()]
            
# #             print(f"✅ Found {len(df)} players with valid IDs")
# #             return df
            
# #         except Exception as e:
# #             print(f"❌ Error scraping players: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             return pd.DataFrame()
    
# #     def scrape_match_results(self, league_url, season_year):
# #         """Scrape match results with scores, xG, and match URLs"""
# #         fixtures_url = league_url.replace('Premier-League-Stats', 'schedule/Premier-League-Scores-and-Fixtures')
# #         print(f"\n⚽ Scraping matches from: {fixtures_url}")
# #         self.random_delay()
        
# #         try:
# #             response = self.scraper.get(fixtures_url)
# #             response.raise_for_status()
# #             soup = BeautifulSoup(response.content, 'html.parser')
            
# #             table = soup.find('table', {'id': re.compile(r'.*sched.*', re.I)})
# #             if not table:
# #                 tables = soup.find_all('table')
# #                 if tables:
# #                     table = tables[0]
            
# #             if not table:
# #                 return pd.DataFrame()
            
# #             df = pd.read_html(str(table))[0]
            
# #             if isinstance(df.columns, pd.MultiIndex):
# #                 df.columns = df.columns.droplevel(0)
            
# #             if 'Score' in df.columns:
# #                 df = df[df['Score'].notna()]
# #                 df = df[df['Score'] != 'Score']
            
# #             match_links = table.find_all('a', href=re.compile(r'/en/matches/'))
            
# #             match_data = []
# #             for link in match_links:
# #                 match_url = self.base_url + link['href']
# #                 fbref_id = self.extract_fbref_id(link['href'])
# #                 match_data.append({
# #                     'match_url': match_url,
# #                     'fbref_id': fbref_id if fbref_id else match_url
# #                 })
            
# #             if match_data:
# #                 for i, row in enumerate(match_data[:len(df)]):
# #                     if i < len(df):
# #                         df.loc[df.index[i], 'match_url'] = row['match_url']
# #                         df.loc[df.index[i], 'fbref_id'] = row['fbref_id']
            
# #             print(f"✅ Found {len(df)} completed matches")
# #             return df
# #         except Exception as e:
# #             print(f"❌ Error scraping matches: {e}")
# #             return pd.DataFrame()
    
# #     def scrape_match_lineups_and_events(self, match_url, match_id):
# #         """Scrape lineups and events from a match page - FIXED"""
# #         print(f"\n📋 Scraping lineups/events: {match_url}")
# #         self.random_delay()
        
# #         try:
# #             response = self.scraper.get(match_url)
# #             response.raise_for_status()
# #             soup = BeautifulSoup(response.content, 'html.parser')
            
# #             lineups = []
# #             events = []
            
# #             # FIX: Better lineup extraction using lineup divs
# #             lineup_divs = soup.find_all('div', {'class': 'lineup'})
            
# #             for lineup_div in lineup_divs:
# #                 # Get team name from preceding h2 or within div
# #                 team_header = lineup_div.find_previous('h2')
# #                 if not team_header:
# #                     continue
                    
# #                 team_name = team_header.text.strip().split(' Lineup')[0].strip()
                
# #                 # Get team_id from database
# #                 cur = self.conn.cursor()
# #                 cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (team_name,))
# #                 team_result = cur.fetchone()
# #                 cur.close()
                
# #                 if not team_result:
# #                     print(f"⚠️  Team not found: {team_name}")
# #                     continue
                
# #                 team_id = team_result[0]
                
# #                 # Find all player links
# #                 player_links = lineup_div.find_all('a', href=re.compile(r'/en/players/'))
                
# #                 for idx, link in enumerate(player_links):
# #                     player_fbref_id = self.extract_fbref_id(link['href'])
# #                     if not player_fbref_id:
# #                         continue
                    
# #                     # Get player_id from database
# #                     cur = self.conn.cursor()
# #                     cur.execute("SELECT player_id FROM PLAYERS WHERE fbref_id = %s", (player_fbref_id,))
# #                     player_result = cur.fetchone()
# #                     cur.close()
                    
# #                     if player_result:
# #                         lineups.append({
# #                             'match_id': match_id,
# #                             'player_id': player_result[0],
# #                             'team_id': team_id,
# #                             'is_starter': idx < 11,
# #                             'minutes_played': 90 if idx < 11 else 0
# #                         })
            
# #             # FIX: Better event extraction
# #             # Look for scorer divs or event tables
# #             scorer_divs = soup.find_all('div', {'class': 'scorer'})
            
# #             for scorer_div in scorer_divs:
# #                 # Extract minute
# #                 minute_span = scorer_div.find('div', {'class': 'min'})
# #                 if not minute_span:
# #                     continue
                
# #                 minute_text = minute_span.text.strip()
# #                 minute_match = re.search(r'(\d+)', minute_text)
# #                 if not minute_match:
# #                     continue
# #                 minute = int(minute_match.group(1))
                
# #                 # Find player link
# #                 player_link = scorer_div.find('a', href=re.compile(r'/en/players/'))
# #                 if not player_link:
# #                     continue
                
# #                 player_fbref_id = self.extract_fbref_id(player_link['href'])
# #                 if not player_fbref_id:
# #                     continue
                
# #                 # Determine event type (goal, own goal, penalty, etc.)
# #                 event_text = scorer_div.text
# #                 if 'own goal' in event_text.lower():
# #                     event_type = 'Own Goal'
# #                 elif 'pen' in event_text.lower():
# #                     event_type = 'Penalty Goal'
# #                 else:
# #                     event_type = 'Goal'
                
# #                 # Get player_id and team_id
# #                 cur = self.conn.cursor()
# #                 cur.execute("""
# #                     SELECT p.player_id, p.team_id 
# #                     FROM PLAYERS p 
# #                     WHERE p.fbref_id = %s
# #                 """, (player_fbref_id,))
# #                 player_result = cur.fetchone()
# #                 cur.close()
                
# #                 if player_result:
# #                     events.append({
# #                         'match_id': match_id,
# #                         'minute': minute,
# #                         'event_type': event_type,
# #                         'player_id': player_result[0],
# #                         'team_id': player_result[1]
# #                     })
            
# #             print(f"✅ Extracted {len(lineups)} lineups, {len(events)} events")
# #             return lineups, events
            
# #         except Exception as e:
# #             print(f"❌ Error scraping match details: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             return [], []
    
# #     def insert_league(self, name, country, tier, fbref_id):
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("""
# #                 INSERT INTO LEAGUES (name, country, tier, fbref_id)
# #                 VALUES (%s, %s, %s, %s)
# #                 ON CONFLICT (fbref_id) DO UPDATE 
# #                 SET name = EXCLUDED.name
# #                 RETURNING league_id
# #             """, (name, country, tier, fbref_id))
# #             league_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             print(f"✅ Inserted league: {name}")
# #             return league_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting league: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_season(self, year, start_date, end_date, league_id):
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("""
# #                 SELECT season_id FROM SEASONS 
# #                 WHERE year = %s AND league_id = %s
# #             """, (year, league_id))
# #             existing = cur.fetchone()
            
# #             if existing:
# #                 season_id = existing[0]
# #                 print(f"ℹ️  Season {year} already exists")
# #             else:
# #                 cur.execute("""
# #                     INSERT INTO SEASONS (year, start_date, end_date, league_id)
# #                     VALUES (%s, %s, %s, %s)
# #                     RETURNING season_id
# #                 """, (year, start_date, end_date, league_id))
# #                 season_id = cur.fetchone()[0]
# #                 print(f"✅ Inserted season: {year}")
            
# #             self.conn.commit()
# #             return season_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting season: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_team(self, name, stadium, city, fbref_id):
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("""
# #                 INSERT INTO TEAMS (name, stadium_name, city, fbref_id)
# #                 VALUES (%s, %s, %s, %s)
# #                 ON CONFLICT (fbref_id) DO UPDATE 
# #                 SET name = EXCLUDED.name
# #                 RETURNING team_id
# #             """, (name, stadium, city, fbref_id))
# #             team_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return team_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting team {name}: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_player(self, full_name, nationality, position, team_id, fbref_id, age=None, shirt_number=None):
# #         """Insert player - FIXED"""
# #         cur = self.conn.cursor()
# #         try:
# #             # Position mapping
# #             pos_map = {
# #                 'GK': 'Goalkeeper', 
# #                 'DF': 'Defender',
# #                 'MF': 'Midfielder', 
# #                 'FW': 'Forward'
# #             }
            
# #             # Clean position
# #             if position and pd.notna(position):
# #                 pos_str = str(position).strip()
# #                 if len(pos_str) >= 2:
# #                     pos_abbr = pos_str[:2].upper()
# #                     position = pos_map.get(pos_abbr, 'Midfielder')
# #                 else:
# #                     position = 'Midfielder'
# #             else:
# #                 position = 'Midfielder'
            
# #             # Calculate DOB from age
# #             dob = None
# #             if age and pd.notna(age) and str(age).strip():
# #                 try:
# #                     age_int = int(float(str(age).split('-')[0]))  # Handle "25-123" format
# #                     current_year = datetime.now().year
# #                     birth_year = current_year - age_int
# #                     dob = f"{birth_year}-01-01"
# #                 except:
# #                     pass
            
# #             # Clean nationality (remove flag emoji or codes)
# #             if nationality and pd.notna(nationality):
# #                 nationality = str(nationality).strip()
# #                 # Remove common patterns like "eng ENG"
# #                 nationality = re.sub(r'\s+[A-Z]{3}$', '', nationality)
# #                 nationality = nationality.strip()[:3].upper() if len(nationality) >= 3 else None
            
# #             # Clean shirt number
# #             if shirt_number and pd.notna(shirt_number):
# #                 try:
# #                     shirt_number = int(float(shirt_number))
# #                 except:
# #                     shirt_number = None
# #             else:
# #                 shirt_number = None
            
# #             cur.execute("""
# #                 INSERT INTO PLAYERS (full_name, nationality, position, team_id, fbref_id, dob, shirt_number)
# #                 VALUES (%s, %s, %s, %s, %s, %s, %s)
# #                 ON CONFLICT (fbref_id) DO UPDATE 
# #                 SET team_id = EXCLUDED.team_id, 
# #                     shirt_number = EXCLUDED.shirt_number,
# #                     full_name = EXCLUDED.full_name,
# #                     position = EXCLUDED.position
# #                 RETURNING player_id
# #             """, (full_name, nationality, position, team_id, fbref_id, dob, shirt_number))
# #             player_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return player_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting player {full_name}: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_match(self, match_date, venue, home_team, away_team, 
# #                     home_score, away_score, home_xg, away_xg, season_id, fbref_id):
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (home_team,))
# #             home_id = cur.fetchone()
# #             cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (away_team,))
# #             away_id = cur.fetchone()
            
# #             if not home_id or not away_id:
# #                 return None
            
# #             cur.execute("""
# #                 INSERT INTO MATCHES 
# #                 (match_date, venue, home_score_final, away_score_final, 
# #                  home_xg, away_xg, season_id, home_team_id, away_team_id, fbref_id)
# #                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
# #                 ON CONFLICT (fbref_id) DO UPDATE
# #                 SET home_score_final = EXCLUDED.home_score_final,
# #                     away_score_final = EXCLUDED.away_score_final
# #                 RETURNING match_id
# #             """, (match_date, venue, home_score, away_score, 
# #                   home_xg, away_xg, season_id, home_id[0], away_id[0], fbref_id))
# #             match_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return match_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting match: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_team_season(self, team_id, season_id, standings_row):
# #         cur = self.conn.cursor()
# #         try:
# #             points = int(standings_row.get('Pts', 0)) if 'Pts' in standings_row and pd.notna(standings_row.get('Pts')) else 0
# #             wins = int(standings_row.get('W', 0)) if 'W' in standings_row and pd.notna(standings_row.get('W')) else 0
# #             draws = int(standings_row.get('D', 0)) if 'D' in standings_row and pd.notna(standings_row.get('D')) else 0
# #             losses = int(standings_row.get('L', 0)) if 'L' in standings_row and pd.notna(standings_row.get('L')) else 0
# #             goals_for = int(standings_row.get('GF', 0)) if 'GF' in standings_row and pd.notna(standings_row.get('GF')) else 0
# #             goals_against = int(standings_row.get('GA', 0)) if 'GA' in standings_row and pd.notna(standings_row.get('GA')) else 0
# #             goal_diff = int(standings_row.get('GD', 0)) if 'GD' in standings_row and pd.notna(standings_row.get('GD')) else 0
            
# #             position = None
# #             for col in ['Rk', 'Pos', 'Position']:
# #                 if col in standings_row and pd.notna(standings_row.get(col)):
# #                     try:
# #                         position = int(standings_row[col])
# #                         break
# #                     except:
# #                         pass
            
# #             cur.execute("""
# #                 INSERT INTO TEAM_SEASONS 
# #                 (team_id, season_id, points, wins, draws, losses, 
# #                  goals_for, goals_against, goal_diff, position)
# #                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
# #                 ON CONFLICT (team_id, season_id) DO UPDATE
# #                 SET points = EXCLUDED.points, wins = EXCLUDED.wins
# #                 RETURNING team_season_id
# #             """, (team_id, season_id, points, wins, draws, losses,
# #                   goals_for, goals_against, goal_diff, position))
            
# #             team_season_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return team_season_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting team_season: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_lineup(self, lineup_data):
# #         """Insert lineup record - FIXED"""
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("""
# #                 INSERT INTO MATCH_LINEUPS 
# #                 (match_id, player_id, team_id, is_starter, minutes_played)
# #                 VALUES (%s, %s, %s, %s, %s)
# #                 RETURNING lineup_id
# #             """, (
# #                 lineup_data['match_id'],
# #                 lineup_data['player_id'],
# #                 lineup_data['team_id'],
# #                 lineup_data['is_starter'],
# #                 lineup_data['minutes_played']
# #             ))
# #             lineup_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return lineup_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting lineup: {e}")
# #             return None
# #         finally:
# #             cur.close()
    
# #     def insert_event(self, event_data):
# #         """Insert match event - FIXED"""
# #         cur = self.conn.cursor()
# #         try:
# #             cur.execute("""
# #                 INSERT INTO MATCH_EVENT 
# #                 (match_id, minute, event_type, player_id, team_id)
# #                 VALUES (%s, %s, %s, %s, %s)
# #                 RETURNING event_id
# #             """, (
# #                 event_data['match_id'],
# #                 event_data['minute'],
# #                 event_data['event_type'],
# #                 event_data['player_id'],
# #                 event_data['team_id']
# #             ))
# #             event_id = cur.fetchone()[0]
# #             self.conn.commit()
# #             return event_id
# #         except Exception as e:
# #             self.conn.rollback()
# #             print(f"❌ Error inserting event: {e}")
# #             return None
# #         finally:
# #             cur.close()


# # def main():
# #     scraper = FBrefScraper()
    
# #     try:
# #         scraper.connect_db()
# #     except:
# #         print("❌ Could not connect to database")
# #         return
    
# #     try:
# #         print("\n" + "="*50)
# #         print("TRUNCATING TABLES")
# #         print("="*50)
# #         scraper.truncate_all_tables()
        
# #         print("\n" + "="*50)
# #         print("INSERTING LEAGUE")
# #         print("="*50)
# #         league_id = scraper.insert_league('Premier League', 'England', 1, '9')
        
# #         if not league_id:
# #             return
        
# #         print("\n" + "="*50)
# #         print("INSERTING SEASON")
# #         print("="*50)
# #         season_id = scraper.insert_season('2024-25', '2024-08-16', '2025-05-25', league_id)
        
# #         if not season_id:
# #             return
        
# #         print("\n" + "="*50)
# #         print("SCRAPING TEAMS")
# #         print("="*50)
# #         league_url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
# #         standings_df = scraper.scrape_league_standings(league_url, '2024-25')
        
# #         if standings_df.empty:
# #             return
        
# #         team_ids = {}
# #         for _, row in tqdm(standings_df.iterrows(), total=len(standings_df), desc="Inserting teams"):
# #             team_name = row['Squad']
# #             fbref_id = row.get('fbref_id', f"team_{team_name.replace(' ', '_').lower()}")
            
# #             team_id = scraper.insert_team(team_name, None, None, fbref_id)
# #             if team_id:
# #                 team_ids[team_name] = {
# #                     'team_id': team_id,
# #                     'team_url': row.get('team_url'),
# #                     'standings_row': row
# #                 }
        
# #         print(f"\n✅ Inserted {len(team_ids)} teams")
        
# #         print("\n" + "="*50)
# #         print("INSERTING TEAM SEASON STATS")
# #         print("="*50)
# #         for team_name, team_info in tqdm(team_ids.items(), desc="Team seasons"):
# #             scraper.insert_team_season(team_info['team_id'], season_id, team_info['standings_row'])
        
# #         print("\n" + "="*50)
# #         print("SCRAPING PLAYERS (ALL TEAMS)")
# #         print("="*50)
# #         player_count = 0
# #         for team_name, team_info in tqdm(list(team_ids.items()), desc="Scraping players"):
# #             if not team_info.get('team_url'):
# #                 continue
            
# #             players_df = scraper.scrape_team_players(team_info['team_url'])
            
# #             if players_df.empty:
# #                 print(f"⚠️  No players found for {team_name}")
# #                 continue
            
# #             print(f"\nProcessing {len(players_df)} players for {team_name}")
# #             print(f"Columns: {players_df.columns.tolist()}")
            
# #             for _, player in players_df.iterrows():
# #                 try:
# #                     player_name = player.get('Player')
# #                     age = player.get('Age')
# #                     position = player.get('Pos')
# #                     nationality = player.get('Nation')
# #                     shirt_number = player.get('#') if '#' in player else player.get('Num')
# #                     fbref_id = player.get('fbref_id')
                    
# #                     if not player_name or not fbref_id:
# #                         continue
                    
# #                     player_id = scraper.insert_player(
# #                         player_name, nationality, position, 
# #                         team_info['team_id'], fbref_id, age, shirt_number
# #                     )
# #                     if player_id:
# #                         player_count += 1
# #                 except Exception as e:
# #                     print(f"⚠️  Failed to insert player: {e}")
# #                     continue
        
# #         print(f"\n✅ Inserted {player_count} players")
        
# #         print("\n" + "="*50)
# #         print("SCRAPING MATCHES")
# #         print("="*50)
# #         matches_df = scraper.scrape_match_results(league_url, '2024-25')
        
# #         if matches_df.empty:
# #             print("⚠️  No matches found")
# #         else:
# #             match_count = 0
# #             match_ids_list = []
            
# #             for _, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Inserting matches"):
# #                 try:
# #                     score = str(row['Score']).split('–')
# #                     if len(score) != 2:
# #                         score = str(row['Score']).split('-')
                    
# #                     home_score = int(score[0].strip()) if len(score) == 2 else None
# #                     away_score = int(score[1].strip()) if len(score) == 2 else None
                    
# #                     home_xg = float(row.get('xG', 0)) if 'xG' in row and pd.notna(row.get('xG')) else None
# #                     away_xg = float(row.get('xG.1', 0)) if 'xG.1' in row and pd.notna(row.get('xG.1')) else None
                    
# #                     fbref_id = row.get('fbref_id', f"match_{row['Date']}_{row['Home']}_{row['Away']}")
                    
# #                     match_id = scraper.insert_match(
# #                         row['Date'], row.get('Venue', ''),
# #                         row['Home'], row['Away'],
# #                         home_score, away_score,
# #                         home_xg, away_xg,
# #                         season_id, fbref_id
# #                     )
                    
# #                     if match_id:
# #                         match_count += 1
# #                         match_ids_list.append({
# #                             'match_id': match_id,
# #                             'match_url': row.get('match_url')
# #                         })
# #                 except Exception as e:
# #                     print(f"⚠️  Skipped match: {e}")
# #                     continue
            
# #             print(f"\n✅ Inserted {match_count} matches")
            
# #             # Scrape lineups and events for first 10 matches
# #             print("\n" + "="*50)
# #             print("SCRAPING LINEUPS & EVENTS (First 10 matches)")
# #             print("="*50)
            
# #             lineup_count = 0
# #             event_count = 0
            
# #             for match_info in tqdm(match_ids_list[:10], desc="Match details"):
# #                 if not match_info.get('match_url'):
# #                     continue
                
# #                 lineups, events = scraper.scrape_match_lineups_and_events(
# #                     match_info['match_url'],
# #                     match_info['match_id']
# #                 )
                
# #                 # Insert lineups
# #                 for lineup in lineups:
# #                     if scraper.insert_lineup(lineup):
# #                         lineup_count += 1
                
# #                 # Insert events
# #                 for event in events:
# #                     if scraper.insert_event(event):
# #                         event_count += 1
            
# #             print(f"\n✅ Inserted {lineup_count} lineup entries")
# #             print(f"✅ Inserted {event_count} match events")
        
# #         print("\n" + "="*50)
# #         print("✅ SCRAPING COMPLETED!")
# #         print("="*50)
        
# #     except Exception as e:
# #         print(f"\n❌ Fatal error: {e}")
# #         import traceback
# #         traceback.print_exc()
# #     finally:
# #         scraper.close_db()


# # if __name__ == '__main__':
# #     main()


# """
# FBref Data Scraper - FIXED VERSION
# """

# import cloudscraper
# from bs4 import BeautifulSoup
# import pandas as pd
# import time
# import psycopg2
# from datetime import datetime
# import os
# from dotenv import load_dotenv
# from tqdm import tqdm
# import random
# import re

# load_dotenv()

# # Increased rate limits to avoid 403
# RATE_LIMIT_MIN = 5
# RATE_LIMIT_MAX = 8

# class FBrefScraper:
#     def __init__(self):
#         self.scraper = cloudscraper.create_scraper(
#             browser={
#                 'browser': 'chrome',
#                 'platform': 'windows',
#                 'mobile': False
#             },
#             delay=10
#         )
#         self.base_url = 'https://fbref.com'
#         self.conn = None
        
#     def connect_db(self):
#         """Connect to PostgreSQL database"""
#         try:
#             self.conn = psycopg2.connect(
#                 host=os.getenv('POSTGRES_HOST', 'localhost'),
#                 port=os.getenv('POSTGRES_PORT', '5432'),
#                 database=os.getenv('POSTGRES_DB', 'spark_db'),
#                 user=os.getenv('POSTGRES_USER', 'spark_user'),
#                 password=os.getenv('POSTGRES_PASSWORD', 'spark_password_2024'),
#                 connect_timeout=10
#             )
#             print("✅ Connected to database")
#         except Exception as e:
#             print(f"❌ Database connection failed: {e}")
#             raise
        
#     def close_db(self):
#         if self.conn:
#             self.conn.close()
#             print("✅ Database connection closed")
    
#     def truncate_all_tables(self):
#         """Truncate all tables before fresh scrape"""
#         cur = self.conn.cursor()
#         try:
#             print("\n🗑️  Truncating all tables...")
#             tables = [
#                 'MATCH_PREDICTIONS', 'MATCH_EVENT', 'MATCH_LINEUPS', 'MATCHES',
#                 'TEAM_SEASONS', 'PLAYER_NICKNAMES', 'STAFF', 'PLAYERS', 
#                 'TEAMS', 'SEASONS', 'LEAGUES'
#             ]
#             for table in tables:
#                 cur.execute(f"TRUNCATE TABLE {table} CASCADE")
#             self.conn.commit()
#             print("✅ All tables truncated")
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error truncating tables: {e}")
#         finally:
#             cur.close()
    
#     def random_delay(self):
#         delay = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
#         time.sleep(delay)
    
#     def extract_fbref_id(self, url):
#         """Extract FBref ID from URL"""
#         match = re.search(r'/([a-f0-9]{8})/', url)
#         if match:
#             return match.group(1)
#         return None
    
#     def scrape_league_standings(self, league_url, season_year):
#         """Scrape league standings with team URLs and FBref IDs"""
#         print(f"\n📊 Scraping standings from: {league_url}")
#         self.random_delay()
        
#         try:
#             response = self.scraper.get(league_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             table = soup.find('table', {'id': re.compile(r'.*standings.*', re.I)})
#             if not table:
#                 tables = soup.find_all('table')
#                 for t in tables:
#                     if 'Squad' in str(t):
#                         table = t
#                         break
            
#             if not table:
#                 print("❌ Could not find standings table")
#                 return pd.DataFrame()
            
#             df = pd.read_html(str(table))[0]
            
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.droplevel(0)
            
#             squad_col = None
#             for col in df.columns:
#                 if 'Squad' in str(col) or 'Team' in str(col):
#                     squad_col = col
#                     break
            
#             if not squad_col:
#                 return pd.DataFrame()
            
#             df = df[df[squad_col] != squad_col]
#             df = df.rename(columns={squad_col: 'Squad'})
            
#             team_links = table.find_all('a', href=re.compile(r'/en/squads/'))
            
#             team_data = []
#             for link in team_links:
#                 team_name = link.text.strip()
#                 team_url = self.base_url + link['href']
#                 fbref_id = self.extract_fbref_id(link['href'])
#                 team_data.append({
#                     'Squad': team_name,
#                     'team_url': team_url,
#                     'fbref_id': fbref_id if fbref_id else team_url
#                 })
            
#             team_df = pd.DataFrame(team_data)
#             df = df.merge(team_df, on='Squad', how='left')
            
#             print(f"✅ Found {len(df)} teams with URLs")
#             return df
            
#         except Exception as e:
#             print(f"❌ Error scraping standings: {e}")
#             return pd.DataFrame()
    
#     def scrape_team_players(self, team_url):
#         """Scrape player data for a specific team - FIXED"""
#         print(f"\n👥 Scraping players from: {team_url}")
#         self.random_delay()
        
#         try:
#             response = self.scraper.get(team_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             table = soup.find('table', {'id': re.compile(r'.*stats_standard.*', re.I)})
#             if not table:
#                 print("❌ Could not find player stats table")
#                 return pd.DataFrame()
            
#             df = pd.read_html(str(table))[0]
            
#             # FIX: Flatten multi-level columns properly
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
            
#             # Remove header rows that repeat
#             df = df[df['Player'] != 'Player']
            
#             # Get player links
#             player_links = table.find_all('a', href=re.compile(r'/en/players/'))
            
#             player_data = []
#             for link in player_links:
#                 player_name = link.text.strip()
#                 player_url = self.base_url + link['href']
#                 fbref_id = self.extract_fbref_id(link['href'])
#                 if fbref_id:  # Only add if we have valid fbref_id
#                     player_data.append({
#                         'Player': player_name,
#                         'player_url': player_url,
#                         'fbref_id': fbref_id
#                     })
            
#             player_df = pd.DataFrame(player_data)
            
#             # Merge on Player name
#             if not player_df.empty and 'Player' in df.columns:
#                 df = df.merge(player_df, on='Player', how='left')
#                 # Remove rows without fbref_id
#                 df = df[df['fbref_id'].notna()]
            
#             print(f"✅ Found {len(df)} players with valid IDs")
#             return df
            
#         except Exception as e:
#             print(f"❌ Error scraping players: {e}")
#             import traceback
#             traceback.print_exc()
#             return pd.DataFrame()
    
#     def scrape_match_results(self, league_url, season_year):
#         """Scrape match results with scores, xG, and match URLs - FIXED"""
#         fixtures_url = league_url.replace('Premier-League-Stats', 'schedule/Premier-League-Scores-and-Fixtures')
#         print(f"\n⚽ Scraping matches from: {fixtures_url}")
#         self.random_delay()
        
#         try:
#             response = self.scraper.get(fixtures_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             table = soup.find('table', {'id': re.compile(r'.*sched.*', re.I)})
#             if not table:
#                 tables = soup.find_all('table')
#                 if tables:
#                     table = tables[0]
            
#             if not table:
#                 return pd.DataFrame()
            
#             df = pd.read_html(str(table))[0]
            
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.droplevel(0)
            
#             if 'Score' in df.columns:
#                 df = df[df['Score'].notna()]
#                 df = df[df['Score'] != 'Score']
            
#             # FIX: Better match URL extraction - look for "Match Report" links
#             match_report_links = table.find_all('a', string=re.compile(r'Match Report', re.I))
            
#             match_data = {}
#             for link in match_report_links:
#                 if '/en/matches/' in link['href']:
#                     match_url = self.base_url + link['href']
#                     fbref_id = self.extract_fbref_id(link['href'])
                    
#                     # Find the row this link belongs to
#                     row = link.find_parent('tr')
#                     if row:
#                         # Get row index from table
#                         all_rows = table.find_all('tr')
#                         row_idx = all_rows.index(row)
#                         match_data[row_idx] = {
#                             'match_url': match_url,
#                             'fbref_id': fbref_id if fbref_id else match_url
#                         }
            
#             # Assign URLs to dataframe
#             df['match_url'] = None
#             df['fbref_id'] = None
            
#             for row_idx, data in match_data.items():
#                 # Adjust for header rows
#                 df_idx = row_idx - 1  # Usually 1 header row
#                 if 0 <= df_idx < len(df):
#                     df.loc[df.index[df_idx], 'match_url'] = data['match_url']
#                     df.loc[df.index[df_idx], 'fbref_id'] = data['fbref_id']
            
#             # Remove rows without match URLs
#             df = df[df['match_url'].notna()]
            
#             print(f"✅ Found {len(df)} completed matches with URLs")
#             return df
#         except Exception as e:
#             print(f"❌ Error scraping matches: {e}")
#             import traceback
#             traceback.print_exc()
#             return pd.DataFrame()
    
#     def scrape_match_lineups_and_events(self, match_url, match_id):
#         """Scrape lineups and events from a match page - COMPLETELY FIXED"""
#         print(f"\n📋 Scraping lineups/events: {match_url}")
#         self.random_delay()
        
#         try:
#             response = self.scraper.get(match_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             lineups = []
#             events = []
            
#             # METHOD 1: Extract lineups from lineup tables
#             # FBref has tables like "Liverpool_d3fd31a7" and "Bournemouth_e4a775cb"
#             lineup_tables = soup.find_all('table', {'id': re.compile(r'.*lineup.*', re.I)})
            
#             if not lineup_tables:
#                 # Alternative: look for tables with class "lineup"
#                 lineup_tables = soup.find_all('table', class_=re.compile(r'lineup', re.I))
            
#             for table in lineup_tables:
#                 # Extract team name from table ID or nearby header
#                 table_id = table.get('id', '')
                
#                 # Get team name from the scorebox or headers
#                 team_name = None
                
#                 # Look backwards for team name in divs/headers
#                 for elem in table.find_all_previous(['div', 'h2', 'strong']):
#                     text = elem.text.strip()
#                     # Try to match against our known teams
#                     cur = self.conn.cursor()
#                     cur.execute("SELECT team_id, name FROM TEAMS")
#                     teams = cur.fetchall()
#                     cur.close()
                    
#                     for team_id, name in teams:
#                         if name.lower() in text.lower() or text.lower() in name.lower():
#                             team_name = name
#                             break
#                     if team_name:
#                         break
                
#                 if not team_name:
#                     continue
                
#                 # Get team_id
#                 cur = self.conn.cursor()
#                 cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (team_name,))
#                 team_result = cur.fetchone()
#                 cur.close()
                
#                 if not team_result:
#                     continue
                
#                 team_id = team_result[0]
                
#                 # Extract player rows from tbody
#                 tbody = table.find('tbody')
#                 if not tbody:
#                     continue
                
#                 rows = tbody.find_all('tr')
                
#                 for idx, row in enumerate(rows):
#                     # Find player link
#                     player_link = row.find('a', href=re.compile(r'/en/players/'))
#                     if not player_link:
#                         continue
                    
#                     player_fbref_id = self.extract_fbref_id(player_link['href'])
#                     if not player_fbref_id:
#                         continue
                    
#                     # Get player_id from database
#                     cur = self.conn.cursor()
#                     cur.execute("SELECT player_id FROM PLAYERS WHERE fbref_id = %s", (player_fbref_id,))
#                     player_result = cur.fetchone()
#                     cur.close()
                    
#                     if player_result:
#                         # Check if starter (usually first 11 rows, or check for specific class)
#                         is_starter = 'starter' in row.get('class', []) or idx < 11
                        
#                         lineups.append({
#                             'match_id': match_id,
#                             'player_id': player_result[0],
#                             'team_id': team_id,
#                             'is_starter': is_starter,
#                             'minutes_played': 90 if is_starter else 0
#                         })
            
#             # METHOD 2: Extract events from scorebox or event listings
#             # Look for goal scorers in the scorebox
#             scorebox = soup.find('div', class_='scorebox')
#             if scorebox:
#                 # Find all scoring events
#                 score_divs = scorebox.find_all('div', class_=re.compile(r'score', re.I))
                
#                 for div in score_divs:
#                     # Look for small tags or divs containing scorer info
#                     scorer_elements = div.find_all(['small', 'div'], recursive=True)
                    
#                     for elem in scorer_elements:
#                         text = elem.text.strip()
                        
#                         # Parse patterns like "Mohamed Salah 23'" or "Salah 23', 45'+2"
#                         matches = re.findall(r'([^,]+?)\s+(\d+)(?:\'|\u2032)', text)
                        
#                         for player_name, minute in matches:
#                             player_name = player_name.strip()
#                             minute = int(minute)
                            
#                             # Find player link nearby
#                             player_link = elem.find('a', href=re.compile(r'/en/players/'))
#                             if not player_link:
#                                 # Search in parent or siblings
#                                 player_link = elem.find_parent().find('a', href=re.compile(r'/en/players/'))
                            
#                             if player_link:
#                                 player_fbref_id = self.extract_fbref_id(player_link['href'])
#                                 if not player_fbref_id:
#                                     continue
                                
#                                 # Get player_id and team_id
#                                 cur = self.conn.cursor()
#                                 cur.execute("""
#                                     SELECT p.player_id, p.team_id 
#                                     FROM PLAYERS p 
#                                     WHERE p.fbref_id = %s
#                                 """, (player_fbref_id,))
#                                 player_result = cur.fetchone()
#                                 cur.close()
                                
#                                 if player_result:
#                                     events.append({
#                                         'match_id': match_id,
#                                         'minute': minute,
#                                         'event_type': 'Goal',
#                                         'player_id': player_result[0],
#                                         'team_id': player_result[1]
#                                     })
            
#             # METHOD 3: Look for shots table for xG events (optional)
#             shots_tables = soup.find_all('table', {'id': re.compile(r'.*shots.*', re.I)})
#             for table in shots_tables:
#                 tbody = table.find('tbody')
#                 if not tbody:
#                     continue
                    
#                 rows = tbody.find_all('tr')
#                 for row in rows:
#                     cells = row.find_all(['td', 'th'])
#                     if len(cells) < 5:
#                         continue
                    
#                     # Extract minute
#                     minute_cell = cells[0].text.strip() if cells else ''
#                     minute_match = re.search(r'(\d+)', minute_cell)
#                     if not minute_match:
#                         continue
#                     minute = int(minute_match.group(1))
                    
#                     # Find player
#                     player_link = row.find('a', href=re.compile(r'/en/players/'))
#                     if not player_link:
#                         continue
                    
#                     player_fbref_id = self.extract_fbref_id(player_link['href'])
#                     if not player_fbref_id:
#                         continue
                    
#                     # Get outcome (goal/shot)
#                     outcome = cells[5].text.strip() if len(cells) > 5 else ''
#                     event_type = 'Goal' if 'goal' in outcome.lower() else 'Shot'
                    
#                     # Get player_id and team_id
#                     cur = self.conn.cursor()
#                     cur.execute("""
#                         SELECT p.player_id, p.team_id 
#                         FROM PLAYERS p 
#                         WHERE p.fbref_id = %s
#                     """, (player_fbref_id,))
#                     player_result = cur.fetchone()
#                     cur.close()
                    
#                     if player_result and event_type == 'Goal':  # Only add goals
#                         events.append({
#                             'match_id': match_id,
#                             'minute': minute,
#                             'event_type': event_type,
#                             'player_id': player_result[0],
#                             'team_id': player_result[1]
#                         })
            
#             print(f"✅ Extracted {len(lineups)} lineups, {len(events)} events")
#             return lineups, events
            
#         except Exception as e:
#             print(f"❌ Error scraping match details: {e}")
#             import traceback
#             traceback.print_exc()
#             return [], []
    
#     def insert_league(self, name, country, tier, fbref_id):
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 INSERT INTO LEAGUES (name, country, tier, fbref_id)
#                 VALUES (%s, %s, %s, %s)
#                 ON CONFLICT (fbref_id) DO UPDATE 
#                 SET name = EXCLUDED.name
#                 RETURNING league_id
#             """, (name, country, tier, fbref_id))
#             league_id = cur.fetchone()[0]
#             self.conn.commit()
#             print(f"✅ Inserted league: {name}")
#             return league_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting league: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_season(self, year, start_date, end_date, league_id):
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 SELECT season_id FROM SEASONS 
#                 WHERE year = %s AND league_id = %s
#             """, (year, league_id))
#             existing = cur.fetchone()
            
#             if existing:
#                 season_id = existing[0]
#                 print(f"ℹ️  Season {year} already exists")
#             else:
#                 cur.execute("""
#                     INSERT INTO SEASONS (year, start_date, end_date, league_id)
#                     VALUES (%s, %s, %s, %s)
#                     RETURNING season_id
#                 """, (year, start_date, end_date, league_id))
#                 season_id = cur.fetchone()[0]
#                 print(f"✅ Inserted season: {year}")
            
#             self.conn.commit()
#             return season_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting season: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_team(self, name, stadium, city, fbref_id):
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 INSERT INTO TEAMS (name, stadium_name, city, fbref_id)
#                 VALUES (%s, %s, %s, %s)
#                 ON CONFLICT (fbref_id) DO UPDATE 
#                 SET name = EXCLUDED.name
#                 RETURNING team_id
#             """, (name, stadium, city, fbref_id))
#             team_id = cur.fetchone()[0]
#             self.conn.commit()
#             return team_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting team {name}: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_player(self, full_name, nationality, position, team_id, fbref_id, age=None, shirt_number=None):
#         """Insert player - FIXED"""
#         cur = self.conn.cursor()
#         try:
#             # Position mapping
#             pos_map = {
#                 'GK': 'Goalkeeper', 
#                 'DF': 'Defender',
#                 'MF': 'Midfielder', 
#                 'FW': 'Forward'
#             }
            
#             # Clean position
#             if position and pd.notna(position):
#                 pos_str = str(position).strip()
#                 if len(pos_str) >= 2:
#                     pos_abbr = pos_str[:2].upper()
#                     position = pos_map.get(pos_abbr, 'Midfielder')
#                 else:
#                     position = 'Midfielder'
#             else:
#                 position = 'Midfielder'
            
#             # Calculate DOB from age
#             dob = None
#             if age and pd.notna(age) and str(age).strip():
#                 try:
#                     age_int = int(float(str(age).split('-')[0]))  # Handle "25-123" format
#                     current_year = datetime.now().year
#                     birth_year = current_year - age_int
#                     dob = f"{birth_year}-01-01"
#                 except:
#                     pass
            
#             # Clean nationality (remove flag emoji or codes)
#             if nationality and pd.notna(nationality):
#                 nationality = str(nationality).strip()
#                 # Remove common patterns like "eng ENG"
#                 nationality = re.sub(r'\s+[A-Z]{3}$', '', nationality)
#                 nationality = nationality.strip()[:3].upper() if len(nationality) >= 3 else None
            
#             # Clean shirt number
#             if shirt_number and pd.notna(shirt_number):
#                 try:
#                     shirt_number = int(float(shirt_number))
#                 except:
#                     shirt_number = None
#             else:
#                 shirt_number = None
            
#             cur.execute("""
#                 INSERT INTO PLAYERS (full_name, nationality, position, team_id, fbref_id, dob, shirt_number)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (fbref_id) DO UPDATE 
#                 SET team_id = EXCLUDED.team_id, 
#                     shirt_number = EXCLUDED.shirt_number,
#                     full_name = EXCLUDED.full_name,
#                     position = EXCLUDED.position
#                 RETURNING player_id
#             """, (full_name, nationality, position, team_id, fbref_id, dob, shirt_number))
#             player_id = cur.fetchone()[0]
#             self.conn.commit()
#             return player_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting player {full_name}: {e}")
#             import traceback
#             traceback.print_exc()
#             return None
#         finally:
#             cur.close()
    
#     def insert_match(self, match_date, venue, home_team, away_team, 
#                     home_score, away_score, home_xg, away_xg, season_id, fbref_id):
#         cur = self.conn.cursor()
#         try:
#             cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (home_team,))
#             home_id = cur.fetchone()
#             cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (away_team,))
#             away_id = cur.fetchone()
            
#             if not home_id or not away_id:
#                 return None
            
#             cur.execute("""
#                 INSERT INTO MATCHES 
#                 (match_date, venue, home_score_final, away_score_final, 
#                  home_xg, away_xg, season_id, home_team_id, away_team_id, fbref_id)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (fbref_id) DO UPDATE
#                 SET home_score_final = EXCLUDED.home_score_final,
#                     away_score_final = EXCLUDED.away_score_final
#                 RETURNING match_id
#             """, (match_date, venue, home_score, away_score, 
#                   home_xg, away_xg, season_id, home_id[0], away_id[0], fbref_id))
#             match_id = cur.fetchone()[0]
#             self.conn.commit()
#             return match_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting match: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_team_season(self, team_id, season_id, standings_row):
#         cur = self.conn.cursor()
#         try:
#             points = int(standings_row.get('Pts', 0)) if 'Pts' in standings_row and pd.notna(standings_row.get('Pts')) else 0
#             wins = int(standings_row.get('W', 0)) if 'W' in standings_row and pd.notna(standings_row.get('W')) else 0
#             draws = int(standings_row.get('D', 0)) if 'D' in standings_row and pd.notna(standings_row.get('D')) else 0
#             losses = int(standings_row.get('L', 0)) if 'L' in standings_row and pd.notna(standings_row.get('L')) else 0
#             goals_for = int(standings_row.get('GF', 0)) if 'GF' in standings_row and pd.notna(standings_row.get('GF')) else 0
#             goals_against = int(standings_row.get('GA', 0)) if 'GA' in standings_row and pd.notna(standings_row.get('GA')) else 0
#             goal_diff = int(standings_row.get('GD', 0)) if 'GD' in standings_row and pd.notna(standings_row.get('GD')) else 0
            
#             position = None
#             for col in ['Rk', 'Pos', 'Position']:
#                 if col in standings_row and pd.notna(standings_row.get(col)):
#                     try:
#                         position = int(standings_row[col])
#                         break
#                     except:
#                         pass
            
#             cur.execute("""
#                 INSERT INTO TEAM_SEASONS 
#                 (team_id, season_id, points, wins, draws, losses, 
#                  goals_for, goals_against, goal_diff, position)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (team_id, season_id) DO UPDATE
#                 SET points = EXCLUDED.points, wins = EXCLUDED.wins
#                 RETURNING team_season_id
#             """, (team_id, season_id, points, wins, draws, losses,
#                   goals_for, goals_against, goal_diff, position))
            
#             team_season_id = cur.fetchone()[0]
#             self.conn.commit()
#             return team_season_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting team_season: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_lineup(self, lineup_data):
#         """Insert lineup record - FIXED"""
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 INSERT INTO MATCH_LINEUPS 
#                 (match_id, player_id, team_id, is_starter, minutes_played)
#                 VALUES (%s, %s, %s, %s, %s)
#                 RETURNING lineup_id
#             """, (
#                 lineup_data['match_id'],
#                 lineup_data['player_id'],
#                 lineup_data['team_id'],
#                 lineup_data['is_starter'],
#                 lineup_data['minutes_played']
#             ))
#             lineup_id = cur.fetchone()[0]
#             self.conn.commit()
#             return lineup_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting lineup: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_event(self, event_data):
#         """Insert match event - FIXED"""
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 INSERT INTO MATCH_EVENT 
#                 (match_id, minute, event_type, player_id, team_id)
#                 VALUES (%s, %s, %s, %s, %s)
#                 RETURNING event_id
#             """, (
#                 event_data['match_id'],
#                 event_data['minute'],
#                 event_data['event_type'],
#                 event_data['player_id'],
#                 event_data['team_id']
#             ))
#             event_id = cur.fetchone()[0]
#             self.conn.commit()
#             return event_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting event: {e}")
#             return None
#         finally:
#             cur.close()


# def main():
#     scraper = FBrefScraper()
    
#     try:
#         scraper.connect_db()
#     except:
#         print("❌ Could not connect to database")
#         return
    
#     try:
#         print("\n" + "="*50)
#         print("TRUNCATING TABLES")
#         print("="*50)
#         scraper.truncate_all_tables()
        
#         print("\n" + "="*50)
#         print("INSERTING LEAGUE")
#         print("="*50)
#         league_id = scraper.insert_league('Premier League', 'England', 1, '9')
        
#         if not league_id:
#             return
        
#         print("\n" + "="*50)
#         print("INSERTING SEASON")
#         print("="*50)
#         season_id = scraper.insert_season('2024-25', '2024-08-16', '2025-05-25', league_id)
        
#         if not season_id:
#             return
        
#         print("\n" + "="*50)
#         print("SCRAPING TEAMS")
#         print("="*50)
#         league_url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
#         standings_df = scraper.scrape_league_standings(league_url, '2024-25')
        
#         if standings_df.empty:
#             return
        
#         team_ids = {}
#         for _, row in tqdm(standings_df.iterrows(), total=len(standings_df), desc="Inserting teams"):
#             team_name = row['Squad']
#             fbref_id = row.get('fbref_id', f"team_{team_name.replace(' ', '_').lower()}")
            
#             team_id = scraper.insert_team(team_name, None, None, fbref_id)
#             if team_id:
#                 team_ids[team_name] = {
#                     'team_id': team_id,
#                     'team_url': row.get('team_url'),
#                     'standings_row': row
#                 }
        
#         print(f"\n✅ Inserted {len(team_ids)} teams")
        
#         print("\n" + "="*50)
#         print("INSERTING TEAM SEASON STATS")
#         print("="*50)
#         for team_name, team_info in tqdm(team_ids.items(), desc="Team seasons"):
#             scraper.insert_team_season(team_info['team_id'], season_id, team_info['standings_row'])
        
#         print("\n" + "="*50)
#         print("SCRAPING PLAYERS (ALL TEAMS)")
#         print("="*50)
#         player_count = 0
#         for team_name, team_info in tqdm(list(team_ids.items()), desc="Scraping players"):
#             if not team_info.get('team_url'):
#                 continue
            
#             players_df = scraper.scrape_team_players(team_info['team_url'])
            
#             if players_df.empty:
#                 print(f"⚠️  No players found for {team_name}")
#                 continue
            
#             print(f"\nProcessing {len(players_df)} players for {team_name}")
#             print(f"Columns: {players_df.columns.tolist()}")
            
#             for _, player in players_df.iterrows():
#                 try:
#                     player_name = player.get('Player')
#                     age = player.get('Age')
#                     position = player.get('Pos')
#                     nationality = player.get('Nation')
#                     shirt_number = player.get('#') if '#' in player else player.get('Num')
#                     fbref_id = player.get('fbref_id')
                    
#                     if not player_name or not fbref_id:
#                         continue
                    
#                     player_id = scraper.insert_player(
#                         player_name, nationality, position, 
#                         team_info['team_id'], fbref_id, age, shirt_number
#                     )
#                     if player_id:
#                         player_count += 1
#                 except Exception as e:
#                     print(f"⚠️  Failed to insert player: {e}")
#                     continue
        
#         print(f"\n✅ Inserted {player_count} players")
        
#         print("\n" + "="*50)
#         print("SCRAPING MATCHES")
#         print("="*50)
#         matches_df = scraper.scrape_match_results(league_url, '2024-25')
        
#         if matches_df.empty:
#             print("⚠️  No matches found")
#         else:
#             match_count = 0
#             match_ids_list = []
            
#             for _, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Inserting matches"):
#                 try:
#                     score = str(row['Score']).split('–')
#                     if len(score) != 2:
#                         score = str(row['Score']).split('-')
                    
#                     home_score = int(score[0].strip()) if len(score) == 2 else None
#                     away_score = int(score[1].strip()) if len(score) == 2 else None
                    
#                     home_xg = float(row.get('xG', 0)) if 'xG' in row and pd.notna(row.get('xG')) else None
#                     away_xg = float(row.get('xG.1', 0)) if 'xG.1' in row and pd.notna(row.get('xG.1')) else None
                    
#                     fbref_id = row.get('fbref_id', f"match_{row['Date']}_{row['Home']}_{row['Away']}")
                    
#                     match_id = scraper.insert_match(
#                         row['Date'], row.get('Venue', ''),
#                         row['Home'], row['Away'],
#                         home_score, away_score,
#                         home_xg, away_xg,
#                         season_id, fbref_id
#                     )
                    
#                     if match_id:
#                         match_count += 1
#                         match_ids_list.append({
#                             'match_id': match_id,
#                             'match_url': row.get('match_url')
#                         })
#                 except Exception as e:
#                     print(f"⚠️  Skipped match: {e}")
#                     continue
            
#             print(f"\n✅ Inserted {match_count} matches")
            
#             # Scrape lineups and events for first 10 matches
#             print("\n" + "="*50)
#             print("SCRAPING LINEUPS & EVENTS (First 10 matches)")
#             print("="*50)
            
#             lineup_count = 0
#             event_count = 0
            
#             for match_info in tqdm(match_ids_list[:10], desc="Match details"):
#                 if not match_info.get('match_url'):
#                     continue
                
#                 lineups, events = scraper.scrape_match_lineups_and_events(
#                     match_info['match_url'],
#                     match_info['match_id']
#                 )
                
#                 # Insert lineups
#                 for lineup in lineups:
#                     if scraper.insert_lineup(lineup):
#                         lineup_count += 1
                
#                 # Insert events
#                 for event in events:
#                     if scraper.insert_event(event):
#                         event_count += 1
            
#             print(f"\n✅ Inserted {lineup_count} lineup entries")
#             print(f"✅ Inserted {event_count} match events")
        
#         print("\n" + "="*50)
#         print("✅ SCRAPING COMPLETED!")
#         print("="*50)
        
#     except Exception as e:
#         print(f"\n❌ Fatal error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         scraper.close_db()


# if __name__ == '__main__':
#     main()


#VERSION 3 

# """
# FBref Data Scraper - FIXED VERSION (Multiple Seasons + Lineups)
# """

# import cloudscraper
# from bs4 import BeautifulSoup
# import pandas as pd
# import time
# import psycopg2
# from datetime import datetime
# import os
# from dotenv import load_dotenv
# from tqdm import tqdm
# import random
# import re

# load_dotenv()

# # Increased rate limits to avoid 403
# RATE_LIMIT_MIN = 5
# RATE_LIMIT_MAX = 8

# class FBrefScraper:
#     def __init__(self):
#         self.scraper = cloudscraper.create_scraper(
#             browser={
#                 'browser': 'chrome',
#                 'platform': 'windows',
#                 'mobile': False
#             },
#             delay=10
#         )
#         self.base_url = 'https://fbref.com'
#         self.conn = None
#         # Cache team names for lineup matching
#         self.team_name_cache = {}
        
#     def connect_db(self):
#         """Connect to PostgreSQL database"""
#         try:
#             self.conn = psycopg2.connect(
#                 host=os.getenv('POSTGRES_HOST', 'localhost'),
#                 port=os.getenv('POSTGRES_PORT', '5432'),
#                 database=os.getenv('POSTGRES_DB', 'spark_db'),
#                 user=os.getenv('POSTGRES_USER', 'spark_user'),
#                 password=os.getenv('POSTGRES_PASSWORD', 'spark_password_2024'),
#                 connect_timeout=10
#             )
#             print("✅ Connected to database")
#         except Exception as e:
#             print(f"❌ Database connection failed: {e}")
#             raise
        
#     def close_db(self):
#         if self.conn:
#             self.conn.close()
#             print("✅ Database connection closed")
    
#     def truncate_all_tables(self):
#         """Truncate all tables before fresh scrape"""
#         cur = self.conn.cursor()
#         try:
#             print("\n🗑️  Truncating all tables...")
#             tables = [
#                 'MATCH_PREDICTIONS', 'MATCH_EVENT', 'MATCH_LINEUPS', 'MATCHES',
#                 'TEAM_SEASONS', 'PLAYER_NICKNAMES', 'STAFF', 'PLAYERS', 
#                 'TEAMS', 'SEASONS', 'LEAGUES'
#             ]
#             for table in tables:
#                 cur.execute(f"TRUNCATE TABLE {table} CASCADE")
#             self.conn.commit()
#             print("✅ All tables truncated")
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error truncating tables: {e}")
#         finally:
#             cur.close()
    
#     def refresh_team_cache(self):
#         """Refresh team name cache from database"""
#         cur = self.conn.cursor()
#         try:
#             cur.execute("SELECT team_id, name, fbref_id FROM TEAMS")
#             teams = cur.fetchall()
#             self.team_name_cache = {
#                 name.lower(): {'team_id': team_id, 'fbref_id': fbref_id}
#                 for team_id, name, fbref_id in teams
#             }
#             print(f"✅ Cached {len(self.team_name_cache)} teams")
#         finally:
#             cur.close()
    
#     def random_delay(self):
#         delay = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
#         time.sleep(delay)
    
#     def extract_fbref_id(self, url):
#         """Extract FBref ID from URL"""
#         match = re.search(r'/([a-f0-9]{8})/', url)
#         if match:
#             return match.group(1)
#         return None
    
#     def scrape_league_standings(self, league_url, season_year):
#         """Scrape league standings with team URLs and FBref IDs"""
#         print(f"\n📊 Scraping standings from: {league_url}")
#         self.random_delay()
        
#         try:
#             response = self.scraper.get(league_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             table = soup.find('table', {'id': re.compile(r'.*standings.*', re.I)})
#             if not table:
#                 tables = soup.find_all('table')
#                 for t in tables:
#                     if 'Squad' in str(t):
#                         table = t
#                         break
            
#             if not table:
#                 print("❌ Could not find standings table")
#                 return pd.DataFrame()
            
#             df = pd.read_html(str(table))[0]
            
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.droplevel(0)
            
#             squad_col = None
#             for col in df.columns:
#                 if 'Squad' in str(col) or 'Team' in str(col):
#                     squad_col = col
#                     break
            
#             if not squad_col:
#                 return pd.DataFrame()
            
#             df = df[df[squad_col] != squad_col]
#             df = df.rename(columns={squad_col: 'Squad'})
            
#             team_links = table.find_all('a', href=re.compile(r'/en/squads/'))
            
#             team_data = []
#             for link in team_links:
#                 team_name = link.text.strip()
#                 team_url = self.base_url + link['href']
#                 fbref_id = self.extract_fbref_id(link['href'])
#                 team_data.append({
#                     'Squad': team_name,
#                     'team_url': team_url,
#                     'fbref_id': fbref_id if fbref_id else team_url
#                 })
            
#             team_df = pd.DataFrame(team_data)
#             df = df.merge(team_df, on='Squad', how='left')
            
#             print(f"✅ Found {len(df)} teams with URLs")
#             return df
            
#         except Exception as e:
#             print(f"❌ Error scraping standings: {e}")
#             return pd.DataFrame()
    
#     def scrape_team_players(self, team_url):
#         """Scrape player data for a specific team"""
#         print(f"\n👥 Scraping players from: {team_url}")
#         self.random_delay()
        
#         try:
#             response = self.scraper.get(team_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             table = soup.find('table', {'id': re.compile(r'.*stats_standard.*', re.I)})
#             if not table:
#                 print("❌ Could not find player stats table")
#                 return pd.DataFrame()
            
#             df = pd.read_html(str(table))[0]
            
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
            
#             df = df[df['Player'] != 'Player']
            
#             player_links = table.find_all('a', href=re.compile(r'/en/players/'))
            
#             player_data = []
#             for link in player_links:
#                 player_name = link.text.strip()
#                 player_url = self.base_url + link['href']
#                 fbref_id = self.extract_fbref_id(link['href'])
#                 if fbref_id:
#                     player_data.append({
#                         'Player': player_name,
#                         'player_url': player_url,
#                         'fbref_id': fbref_id
#                     })
            
#             player_df = pd.DataFrame(player_data)
            
#             if not player_df.empty and 'Player' in df.columns:
#                 df = df.merge(player_df, on='Player', how='left')
#                 df = df[df['fbref_id'].notna()]
            
#             print(f"✅ Found {len(df)} players with valid IDs")
#             return df
            
#         except Exception as e:
#             print(f"❌ Error scraping players: {e}")
#             import traceback
#             traceback.print_exc()
#             return pd.DataFrame()
    
#     def scrape_match_results(self, league_url, season_year):
#         """Scrape match results with scores, xG, and match URLs"""
#         fixtures_url = league_url.replace('Premier-League-Stats', 'schedule/Premier-League-Scores-and-Fixtures')
#         print(f"\n⚽ Scraping matches from: {fixtures_url}")
#         self.random_delay()
        
#         try:
#             response = self.scraper.get(fixtures_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             table = soup.find('table', {'id': re.compile(r'.*sched.*', re.I)})
#             if not table:
#                 tables = soup.find_all('table')
#                 if tables:
#                     table = tables[0]
            
#             if not table:
#                 return pd.DataFrame()
            
#             df = pd.read_html(str(table))[0]
            
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.droplevel(0)
            
#             if 'Score' in df.columns:
#                 df = df[df['Score'].notna()]
#                 df = df[df['Score'] != 'Score']
            
#             match_report_links = table.find_all('a', string=re.compile(r'Match Report', re.I))
            
#             match_data = {}
#             for link in match_report_links:
#                 if '/en/matches/' in link['href']:
#                     match_url = self.base_url + link['href']
#                     fbref_id = self.extract_fbref_id(link['href'])
                    
#                     row = link.find_parent('tr')
#                     if row:
#                         all_rows = table.find_all('tr')
#                         row_idx = all_rows.index(row)
#                         match_data[row_idx] = {
#                             'match_url': match_url,
#                             'fbref_id': fbref_id if fbref_id else match_url
#                         }
            
#             df['match_url'] = None
#             df['fbref_id'] = None
            
#             for row_idx, data in match_data.items():
#                 df_idx = row_idx - 1
#                 if 0 <= df_idx < len(df):
#                     df.loc[df.index[df_idx], 'match_url'] = data['match_url']
#                     df.loc[df.index[df_idx], 'fbref_id'] = data['fbref_id']
            
#             df = df[df['match_url'].notna()]
            
#             print(f"✅ Found {len(df)} completed matches with URLs")
#             return df
#         except Exception as e:
#             print(f"❌ Error scraping matches: {e}")
#             import traceback
#             traceback.print_exc()
#             return pd.DataFrame()
    
#     def scrape_match_lineups_and_events(self, match_url, match_id):
#         """Scrape lineups and events from a match page - COMPLETELY FIXED"""
#         print(f"\n📋 Scraping lineups/events: {match_url}")
#         self.random_delay()
        
#         try:
#             response = self.scraper.get(match_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             lineups = []
#             events = []
            
#             # IMPROVED: Get team names from scorebox first
#             scorebox = soup.find('div', class_='scorebox')
#             team_names = []
            
#             if scorebox:
#                 team_divs = scorebox.find_all('div', itemprop='performer')
#                 for div in team_divs:
#                     team_link = div.find('a', itemprop='name')
#                     if team_link:
#                         team_name = team_link.text.strip()
#                         team_names.append(team_name)
            
#             print(f"Found teams in scorebox: {team_names}")
            
#             # IMPROVED: Find lineup sections by looking for team headers
#             lineup_divs = soup.find_all('div', class_=re.compile(r'lineup', re.I))
            
#             for lineup_div in lineup_divs:
#                 # Find team name in header
#                 team_name = None
#                 header = lineup_div.find_previous(['h2', 'strong', 'div'], class_=re.compile(r'th|header'))
#                 if header:
#                     header_text = header.text.strip()
#                     for tn in team_names:
#                         if tn.lower() in header_text.lower():
#                             team_name = tn
#                             break
                
#                 if not team_name and team_names:
#                     # Try to determine from position in page
#                     lineup_index = lineup_divs.index(lineup_div)
#                     if lineup_index < len(team_names):
#                         team_name = team_names[lineup_index]
                
#                 if not team_name:
#                     continue
                
#                 # Match team name from cache (case-insensitive)
#                 team_info = None
#                 for cached_name, info in self.team_name_cache.items():
#                     if cached_name in team_name.lower() or team_name.lower() in cached_name:
#                         team_info = info
#                         break
                
#                 if not team_info:
#                     print(f"⚠️  Team '{team_name}' not found in cache")
#                     continue
                
#                 team_id = team_info['team_id']
#                 print(f"✅ Processing lineup for {team_name} (team_id: {team_id})")
                
#                 # Find table within this lineup div
#                 table = lineup_div.find('table')
#                 if not table:
#                     continue
                
#                 tbody = table.find('tbody')
#                 if not tbody:
#                     continue
                
#                 rows = tbody.find_all('tr')
                
#                 for idx, row in enumerate(rows):
#                     player_link = row.find('a', href=re.compile(r'/en/players/'))
#                     if not player_link:
#                         continue
                    
#                     player_fbref_id = self.extract_fbref_id(player_link['href'])
#                     if not player_fbref_id:
#                         continue
                    
#                     # Get player_id from database
#                     cur = self.conn.cursor()
#                     cur.execute("SELECT player_id FROM PLAYERS WHERE fbref_id = %s", (player_fbref_id,))
#                     player_result = cur.fetchone()
#                     cur.close()
                    
#                     if player_result:
#                         # Determine if starter (first 11 rows typically)
#                         is_starter = idx < 11
                        
#                         lineups.append({
#                             'match_id': match_id,
#                             'player_id': player_result[0],
#                             'team_id': team_id,
#                             'is_starter': is_starter,
#                             'minutes_played': 90 if is_starter else 0
#                         })
            
#             # Extract events from scorebox
#             if scorebox:
#                 score_divs = scorebox.find_all('div', class_=re.compile(r'score', re.I))
                
#                 for div in score_divs:
#                     scorer_elements = div.find_all(['small', 'div'], recursive=True)
                    
#                     for elem in scorer_elements:
#                         text = elem.text.strip()
#                         matches = re.findall(r'([^,]+?)\s+(\d+)(?:\'|\u2032)', text)
                        
#                         for player_name, minute in matches:
#                             player_name = player_name.strip()
#                             minute = int(minute)
                            
#                             player_link = elem.find('a', href=re.compile(r'/en/players/'))
#                             if not player_link:
#                                 player_link = elem.find_parent().find('a', href=re.compile(r'/en/players/'))
                            
#                             if player_link:
#                                 player_fbref_id = self.extract_fbref_id(player_link['href'])
#                                 if not player_fbref_id:
#                                     continue
                                
#                                 cur = self.conn.cursor()
#                                 cur.execute("""
#                                     SELECT p.player_id, p.team_id 
#                                     FROM PLAYERS p 
#                                     WHERE p.fbref_id = %s
#                                 """, (player_fbref_id,))
#                                 player_result = cur.fetchone()
#                                 cur.close()
                                
#                                 if player_result:
#                                     events.append({
#                                         'match_id': match_id,
#                                         'minute': minute,
#                                         'event_type': 'Goal',
#                                         'player_id': player_result[0],
#                                         'team_id': player_result[1]
#                                     })
            
#             print(f"✅ Extracted {len(lineups)} lineups, {len(events)} events")
#             return lineups, events
            
#         except Exception as e:
#             print(f"❌ Error scraping match details: {e}")
#             import traceback
#             traceback.print_exc()
#             return [], []
    
#     def insert_league(self, name, country, tier, fbref_id):
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 INSERT INTO LEAGUES (name, country, tier, fbref_id)
#                 VALUES (%s, %s, %s, %s)
#                 ON CONFLICT (fbref_id) DO UPDATE 
#                 SET name = EXCLUDED.name
#                 RETURNING league_id
#             """, (name, country, tier, fbref_id))
#             league_id = cur.fetchone()[0]
#             self.conn.commit()
#             print(f"✅ Inserted league: {name}")
#             return league_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting league: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_season(self, year, start_date, end_date, league_id):
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 SELECT season_id FROM SEASONS 
#                 WHERE year = %s AND league_id = %s
#             """, (year, league_id))
#             existing = cur.fetchone()
            
#             if existing:
#                 season_id = existing[0]
#                 print(f"ℹ️  Season {year} already exists")
#             else:
#                 cur.execute("""
#                     INSERT INTO SEASONS (year, start_date, end_date, league_id)
#                     VALUES (%s, %s, %s, %s)
#                     RETURNING season_id
#                 """, (year, start_date, end_date, league_id))
#                 season_id = cur.fetchone()[0]
#                 print(f"✅ Inserted season: {year}")
            
#             self.conn.commit()
#             return season_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting season: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_team(self, name, stadium, city, fbref_id):
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 INSERT INTO TEAMS (name, stadium_name, city, fbref_id)
#                 VALUES (%s, %s, %s, %s)
#                 ON CONFLICT (fbref_id) DO UPDATE 
#                 SET name = EXCLUDED.name
#                 RETURNING team_id
#             """, (name, stadium, city, fbref_id))
#             team_id = cur.fetchone()[0]
#             self.conn.commit()
#             return team_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting team {name}: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_player(self, full_name, nationality, position, team_id, fbref_id, age=None, shirt_number=None):
#         """Insert player"""
#         cur = self.conn.cursor()
#         try:
#             pos_map = {
#                 'GK': 'Goalkeeper', 
#                 'DF': 'Defender',
#                 'MF': 'Midfielder', 
#                 'FW': 'Forward'
#             }
            
#             if position and pd.notna(position):
#                 pos_str = str(position).strip()
#                 if len(pos_str) >= 2:
#                     pos_abbr = pos_str[:2].upper()
#                     position = pos_map.get(pos_abbr, 'Midfielder')
#                 else:
#                     position = 'Midfielder'
#             else:
#                 position = 'Midfielder'
            
#             dob = None
#             if age and pd.notna(age) and str(age).strip():
#                 try:
#                     age_int = int(float(str(age).split('-')[0]))
#                     current_year = datetime.now().year
#                     birth_year = current_year - age_int
#                     dob = f"{birth_year}-01-01"
#                 except:
#                     pass
            
#             if nationality and pd.notna(nationality):
#                 nationality = str(nationality).strip()
#                 nationality = re.sub(r'\s+[A-Z]{3}$', '', nationality)
#                 nationality = nationality.strip()[:3].upper() if len(nationality) >= 3 else None
            
#             if shirt_number and pd.notna(shirt_number):
#                 try:
#                     shirt_number = int(float(shirt_number))
#                 except:
#                     shirt_number = None
#             else:
#                 shirt_number = None
            
#             cur.execute("""
#                 INSERT INTO PLAYERS (full_name, nationality, position, team_id, fbref_id, dob, shirt_number)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (fbref_id) DO UPDATE 
#                 SET team_id = EXCLUDED.team_id, 
#                     shirt_number = EXCLUDED.shirt_number,
#                     full_name = EXCLUDED.full_name,
#                     position = EXCLUDED.position
#                 RETURNING player_id
#             """, (full_name, nationality, position, team_id, fbref_id, dob, shirt_number))
#             player_id = cur.fetchone()[0]
#             self.conn.commit()
#             return player_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting player {full_name}: {e}")
#             import traceback
#             traceback.print_exc()
#             return None
#         finally:
#             cur.close()
    
#     def insert_match(self, match_date, venue, home_team, away_team, 
#                     home_score, away_score, home_xg, away_xg, season_id, fbref_id):
#         cur = self.conn.cursor()
#         try:
#             cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (home_team,))
#             home_id = cur.fetchone()
#             cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (away_team,))
#             away_id = cur.fetchone()
            
#             if not home_id or not away_id:
#                 return None
            
#             cur.execute("""
#                 INSERT INTO MATCHES 
#                 (match_date, venue, home_score_final, away_score_final, 
#                  home_xg, away_xg, season_id, home_team_id, away_team_id, fbref_id)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (fbref_id) DO UPDATE
#                 SET home_score_final = EXCLUDED.home_score_final,
#                     away_score_final = EXCLUDED.away_score_final
#                 RETURNING match_id
#             """, (match_date, venue, home_score, away_score, 
#                   home_xg, away_xg, season_id, home_id[0], away_id[0], fbref_id))
#             match_id = cur.fetchone()[0]
#             self.conn.commit()
#             return match_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting match: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_team_season(self, team_id, season_id, standings_row):
#         cur = self.conn.cursor()
#         try:
#             points = int(standings_row.get('Pts', 0)) if 'Pts' in standings_row and pd.notna(standings_row.get('Pts')) else 0
#             wins = int(standings_row.get('W', 0)) if 'W' in standings_row and pd.notna(standings_row.get('W')) else 0
#             draws = int(standings_row.get('D', 0)) if 'D' in standings_row and pd.notna(standings_row.get('D')) else 0
#             losses = int(standings_row.get('L', 0)) if 'L' in standings_row and pd.notna(standings_row.get('L')) else 0
#             goals_for = int(standings_row.get('GF', 0)) if 'GF' in standings_row and pd.notna(standings_row.get('GF')) else 0
#             goals_against = int(standings_row.get('GA', 0)) if 'GA' in standings_row and pd.notna(standings_row.get('GA')) else 0
#             goal_diff = int(standings_row.get('GD', 0)) if 'GD' in standings_row and pd.notna(standings_row.get('GD')) else 0
            
#             position = None
#             for col in ['Rk', 'Pos', 'Position']:
#                 if col in standings_row and pd.notna(standings_row.get(col)):
#                     try:
#                         position = int(standings_row[col])
#                         break
#                     except:
#                         pass
            
#             cur.execute("""
#                 INSERT INTO TEAM_SEASONS 
#                 (team_id, season_id, points, wins, draws, losses, 
#                  goals_for, goals_against, goal_diff, position)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (team_id, season_id) DO UPDATE
#                 SET points = EXCLUDED.points, wins = EXCLUDED.wins
#                 RETURNING team_season_id
#             """, (team_id, season_id, points, wins, draws, losses,
#                   goals_for, goals_against, goal_diff, position))
            
#             team_season_id = cur.fetchone()[0]
#             self.conn.commit()
#             return team_season_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting team_season: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_lineup(self, lineup_data):
#         """Insert lineup record - FIXED with conflict handling"""
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 INSERT INTO MATCH_LINEUPS 
#                 (match_id, player_id, team_id, is_starter, minutes_played)
#                 VALUES (%s, %s, %s, %s, %s)
#                 ON CONFLICT (match_id, player_id) DO UPDATE
#                 SET is_starter = EXCLUDED.is_starter,
#                     minutes_played = EXCLUDED.minutes_played
#                 RETURNING lineup_id
#             """, (
#                 lineup_data['match_id'],
#                 lineup_data['player_id'],
#                 lineup_data['team_id'],
#                 lineup_data['is_starter'],
#                 lineup_data['minutes_played']
#             ))
#             lineup_id = cur.fetchone()[0]
#             self.conn.commit()
#             return lineup_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting lineup: {e}")
#             import traceback
#             traceback.print_exc()
#             return None
#         finally:
#             cur.close()
    
#     def insert_event(self, event_data):
#         """Insert match event"""
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 INSERT INTO MATCH_EVENT 
#                 (match_id, minute, event_type, player_id, team_id)
#                 VALUES (%s, %s, %s, %s, %s)
#                 RETURNING event_id
#             """, (
#                 event_data['match_id'],
#                 event_data['minute'],
#                 event_data['event_type'],
#                 event_data['player_id'],
#                 event_data['team_id']
#             ))
#             event_id = cur.fetchone()[0]
#             self.conn.commit()
#             return event_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting event: {e}")
#             return None
#         finally:
#             cur.close()


# def main():
#     scraper = FBrefScraper()
    
#     try:
#         scraper.connect_db()
#     except:
#         print("❌ Could not connect to database")
#         return
    
#     try:
#         print("\n" + "="*50)
#         print("TRUNCATING TABLES")
#         print("="*50)
#         scraper.truncate_all_tables()
        
#         print("\n" + "="*50)
#         print("INSERTING LEAGUE")
#         print("="*50)
#         league_id = scraper.insert_league('Premier League', 'England', 1, '9')
        
#         if not league_id:
#             return
        
#         # MULTIPLE SEASONS - Scrape last 3 seasons
#         seasons_config = [
#             {
#                 'year': '2024-25',
#                 'start': '2024-08-16',
#                 'end': '2025-05-25',
#                 'url': 'https://fbref.com/en/comps/9/Premier-League-Stats'
#             },
#             {
#                 'year': '2023-24',
#                 'start': '2023-08-11',
#                 'end': '2024-05-19',
#                 'url': 'https://fbref.com/en/comps/9/2023-2024/2023-2024-Premier-League-Stats'
#             },
#             {
#                 'year': '2022-23',
#                 'start': '2022-08-05',
#                 'end': '2023-05-28',
#                 'url': 'https://fbref.com/en/comps/9/2022-2023/2022-2023-Premier-League-Stats'
#             }
#         ]
        
#         for season_config in seasons_config:
#             print("\n" + "="*70)
#             print(f"PROCESSING SEASON: {season_config['year']}")
#             print("="*70)
            
#             season_id = scraper.insert_season(
#                 season_config['year'], 
#                 season_config['start'], 
#                 season_config['end'], 
#                 league_id
#             )
            
#             if not season_id:
#                 print(f"❌ Failed to create season {season_config['year']}, skipping...")
#                 continue
            
#             league_url = season_config['url']
            
#             print("\n" + "="*50)
#             print(f"SCRAPING TEAMS - {season_config['year']}")
#             print("="*50)
#             standings_df = scraper.scrape_league_standings(league_url, season_config['year'])
            
#             if standings_df.empty:
#                 print(f"⚠️  No standings found for {season_config['year']}")
#                 continue
            
#             team_ids = {}
#             for _, row in tqdm(standings_df.iterrows(), total=len(standings_df), desc="Inserting teams"):
#                 team_name = row['Squad']
#                 fbref_id = row.get('fbref_id', f"team_{team_name.replace(' ', '_').lower()}")
                
#                 team_id = scraper.insert_team(team_name, None, None, fbref_id)
#                 if team_id:
#                     team_ids[team_name] = {
#                         'team_id': team_id,
#                         'team_url': row.get('team_url'),
#                         'standings_row': row
#                     }
            
#             print(f"\n✅ Inserted {len(team_ids)} teams for {season_config['year']}")
            
#             # Refresh team cache for lineup matching
#             scraper.refresh_team_cache()
            
#             print("\n" + "="*50)
#             print(f"INSERTING TEAM SEASON STATS - {season_config['year']}")
#             print("="*50)
#             for team_name, team_info in tqdm(team_ids.items(), desc="Team seasons"):
#                 scraper.insert_team_season(team_info['team_id'], season_id, team_info['standings_row'])
            
#             print("\n" + "="*50)
#             print(f"SCRAPING PLAYERS - {season_config['year']}")
#             print("="*50)
#             player_count = 0
#             for team_name, team_info in tqdm(list(team_ids.items()), desc="Scraping players"):
#                 if not team_info.get('team_url'):
#                     continue
                
#                 players_df = scraper.scrape_team_players(team_info['team_url'])
                
#                 if players_df.empty:
#                     print(f"⚠️  No players found for {team_name}")
#                     continue
                
#                 for _, player in players_df.iterrows():
#                     try:
#                         player_name = player.get('Player')
#                         age = player.get('Age')
#                         position = player.get('Pos')
#                         nationality = player.get('Nation')
#                         shirt_number = player.get('#') if '#' in player else player.get('Num')
#                         fbref_id = player.get('fbref_id')
                        
#                         if not player_name or not fbref_id:
#                             continue
                        
#                         player_id = scraper.insert_player(
#                             player_name, nationality, position, 
#                             team_info['team_id'], fbref_id, age, shirt_number
#                         )
#                         if player_id:
#                             player_count += 1
#                     except Exception as e:
#                         print(f"⚠️  Failed to insert player: {e}")
#                         continue
            
#             print(f"\n✅ Inserted {player_count} players for {season_config['year']}")
            
#             print("\n" + "="*50)
#             print(f"SCRAPING MATCHES - {season_config['year']}")
#             print("="*50)
#             matches_df = scraper.scrape_match_results(league_url, season_config['year'])
            
#             if matches_df.empty:
#                 print(f"⚠️  No matches found for {season_config['year']}")
#                 continue
            
#             match_count = 0
#             match_ids_list = []
            
#             for _, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Inserting matches"):
#                 try:
#                     score = str(row['Score']).split('–')
#                     if len(score) != 2:
#                         score = str(row['Score']).split('-')
                    
#                     home_score = int(score[0].strip()) if len(score) == 2 else None
#                     away_score = int(score[1].strip()) if len(score) == 2 else None
                    
#                     home_xg = float(row.get('xG', 0)) if 'xG' in row and pd.notna(row.get('xG')) else None
#                     away_xg = float(row.get('xG.1', 0)) if 'xG.1' in row and pd.notna(row.get('xG.1')) else None
                    
#                     fbref_id = row.get('fbref_id', f"match_{row['Date']}_{row['Home']}_{row['Away']}")
                    
#                     match_id = scraper.insert_match(
#                         row['Date'], row.get('Venue', ''),
#                         row['Home'], row['Away'],
#                         home_score, away_score,
#                         home_xg, away_xg,
#                         season_id, fbref_id
#                     )
                    
#                     if match_id:
#                         match_count += 1
#                         match_ids_list.append({
#                             'match_id': match_id,
#                             'match_url': row.get('match_url')
#                         })
#                 except Exception as e:
#                     print(f"⚠️  Skipped match: {e}")
#                     continue
            
#             print(f"\n✅ Inserted {match_count} matches for {season_config['year']}")
            
#             # Scrape lineups and events for ALL matches (not just first 10)
#             print("\n" + "="*50)
#             print(f"SCRAPING LINEUPS & EVENTS - {season_config['year']} (ALL MATCHES)")
#             print("="*50)
            
#             lineup_count = 0
#             event_count = 0
            
#             # Process ALL matches
#             for match_info in tqdm(match_ids_list, desc="Match details"):
#                 if not match_info.get('match_url'):
#                     continue
                
#                 lineups, events = scraper.scrape_match_lineups_and_events(
#                     match_info['match_url'],
#                     match_info['match_id']
#                 )
                
#                 # Insert lineups
#                 for lineup in lineups:
#                     if scraper.insert_lineup(lineup):
#                         lineup_count += 1
                
#                 # Insert events
#                 for event in events:
#                     if scraper.insert_event(event):
#                         event_count += 1
            
#             print(f"\n✅ Season {season_config['year']} Summary:")
#             print(f"   - Teams: {len(team_ids)}")
#             print(f"   - Players: {player_count}")
#             print(f"   - Matches: {match_count}")
#             print(f"   - Lineups: {lineup_count}")
#             print(f"   - Events: {event_count}")
        
#         print("\n" + "="*70)
#         print("✅ ALL SEASONS SCRAPING COMPLETED!")
#         print("="*70)
        
#     except Exception as e:
#         print(f"\n❌ Fatal error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         scraper.close_db()


# if __name__ == '__main__':
#     main()

"""
FBref Data Scraper - FIXED VERSION (Multiple Seasons + Lineups)
"""

import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import time
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv
from tqdm import tqdm
import random
import re

load_dotenv()

# Increased rate limits to avoid 403
RATE_LIMIT_MIN = 5
RATE_LIMIT_MAX = 8

class FBrefScraper:
    def __init__(self):
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'mobile': False
            },
            delay=10
        )
        self.base_url = 'https://fbref.com'
        self.conn = None
        # Cache team names for lineup matching
        self.team_name_cache = {}
        
    def connect_db(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                database=os.getenv('POSTGRES_DB', 'spark_db'),
                user=os.getenv('POSTGRES_USER', 'spark_user'),
                password=os.getenv('POSTGRES_PASSWORD', 'spark_password_2024'),
                connect_timeout=10
            )
            print("✅ Connected to database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
        
    def close_db(self):
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")
    
    def truncate_all_tables(self):
        """Truncate all tables before fresh scrape"""
        cur = self.conn.cursor()
        try:
            print("\n🗑️  Truncating all tables...")
            tables = [
                'MATCH_PREDICTIONS', 'MATCH_EVENT', 'MATCH_LINEUPS', 'MATCHES',
                'TEAM_SEASONS', 'PLAYER_NICKNAMES', 'STAFF', 'PLAYERS', 
                'TEAMS', 'SEASONS', 'LEAGUES'
            ]
            for table in tables:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE")
            self.conn.commit()
            print("✅ All tables truncated")
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error truncating tables: {e}")
        finally:
            cur.close()
    
    def refresh_team_cache(self):
        """Refresh team name cache from database"""
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT team_id, name, fbref_id FROM TEAMS")
            teams = cur.fetchall()
            self.team_name_cache = {
                name.lower(): {'team_id': team_id, 'fbref_id': fbref_id}
                for team_id, name, fbref_id in teams
            }
            print(f"✅ Cached {len(self.team_name_cache)} teams")
        finally:
            cur.close()
    
    def random_delay(self):
        delay = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
        time.sleep(delay)
    
    def extract_fbref_id(self, url):
        """Extract FBref ID from URL"""
        match = re.search(r'/([a-f0-9]{8})/', url)
        if match:
            return match.group(1)
        return None
    
    def scrape_league_standings(self, league_url, season_year):
        """Scrape league standings with team URLs and FBref IDs"""
        print(f"\n📊 Scraping standings from: {league_url}")
        self.random_delay()
        
        try:
            response = self.scraper.get(league_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            table = soup.find('table', {'id': re.compile(r'.*standings.*', re.I)})
            if not table:
                tables = soup.find_all('table')
                for t in tables:
                    if 'Squad' in str(t):
                        table = t
                        break
            
            if not table:
                print("❌ Could not find standings table")
                return pd.DataFrame()
            
            df = pd.read_html(str(table))[0]
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            squad_col = None
            for col in df.columns:
                if 'Squad' in str(col) or 'Team' in str(col):
                    squad_col = col
                    break
            
            if not squad_col:
                return pd.DataFrame()
            
            df = df[df[squad_col] != squad_col]
            df = df.rename(columns={squad_col: 'Squad'})
            
            team_links = table.find_all('a', href=re.compile(r'/en/squads/'))
            
            team_data = []
            for link in team_links:
                team_name = link.text.strip()
                team_url = self.base_url + link['href']
                fbref_id = self.extract_fbref_id(link['href'])
                team_data.append({
                    'Squad': team_name,
                    'team_url': team_url,
                    'fbref_id': fbref_id if fbref_id else team_url
                })
            
            team_df = pd.DataFrame(team_data)
            df = df.merge(team_df, on='Squad', how='left')
            
            print(f"✅ Found {len(df)} teams with URLs")
            return df
            
        except Exception as e:
            print(f"❌ Error scraping standings: {e}")
            return pd.DataFrame()
    
    def scrape_team_players(self, team_url):
        """Scrape player data for a specific team"""
        print(f"\n👥 Scraping players from: {team_url}")
        self.random_delay()
        
        try:
            response = self.scraper.get(team_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            table = soup.find('table', {'id': re.compile(r'.*stats_standard.*', re.I)})
            if not table:
                print("❌ Could not find player stats table")
                return pd.DataFrame()
            
            df = pd.read_html(str(table))[0]
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
            
            df = df[df['Player'] != 'Player']
            
            player_links = table.find_all('a', href=re.compile(r'/en/players/'))
            
            player_data = []
            for link in player_links:
                player_name = link.text.strip()
                player_url = self.base_url + link['href']
                fbref_id = self.extract_fbref_id(link['href'])
                if fbref_id:
                    player_data.append({
                        'Player': player_name,
                        'player_url': player_url,
                        'fbref_id': fbref_id
                    })
            
            player_df = pd.DataFrame(player_data)
            
            if not player_df.empty and 'Player' in df.columns:
                df = df.merge(player_df, on='Player', how='left')
                df = df[df['fbref_id'].notna()]
            
            print(f"✅ Found {len(df)} players with valid IDs")
            return df
            
        except Exception as e:
            print(f"❌ Error scraping players: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def scrape_match_results(self, league_url, season_year):
        """Scrape match results with scores, xG, and match URLs"""
        fixtures_url = league_url.replace('Premier-League-Stats', 'schedule/Premier-League-Scores-and-Fixtures')
        print(f"\n⚽ Scraping matches from: {fixtures_url}")
        self.random_delay()
        
        try:
            response = self.scraper.get(fixtures_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            table = soup.find('table', {'id': re.compile(r'.*sched.*', re.I)})
            if not table:
                tables = soup.find_all('table')
                if tables:
                    table = tables[0]
            
            if not table:
                return pd.DataFrame()
            
            df = pd.read_html(str(table))[0]
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            if 'Score' in df.columns:
                df = df[df['Score'].notna()]
                df = df[df['Score'] != 'Score']
            
            match_report_links = table.find_all('a', string=re.compile(r'Match Report', re.I))
            
            match_data = {}
            for link in match_report_links:
                if '/en/matches/' in link['href']:
                    match_url = self.base_url + link['href']
                    fbref_id = self.extract_fbref_id(link['href'])
                    
                    row = link.find_parent('tr')
                    if row:
                        all_rows = table.find_all('tr')
                        row_idx = all_rows.index(row)
                        match_data[row_idx] = {
                            'match_url': match_url,
                            'fbref_id': fbref_id if fbref_id else match_url
                        }
            
            df['match_url'] = None
            df['fbref_id'] = None
            
            for row_idx, data in match_data.items():
                df_idx = row_idx - 1
                if 0 <= df_idx < len(df):
                    df.loc[df.index[df_idx], 'match_url'] = data['match_url']
                    df.loc[df.index[df_idx], 'fbref_id'] = data['fbref_id']
            
            df = df[df['match_url'].notna()]
            
            print(f"✅ Found {len(df)} completed matches with URLs")
            return df
        except Exception as e:
            print(f"❌ Error scraping matches: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def scrape_match_lineups_and_events(self, match_url, match_id):
        """Scrape lineups and events from a match page - COMPLETELY FIXED"""
        print(f"\n📋 Scraping lineups/events: {match_url}")
        self.random_delay()
        
        try:
            response = self.scraper.get(match_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            lineups = []
            events = []
            
            # FIX: Get teams directly from database using match_id
            cur = self.conn.cursor()
            cur.execute("""
                SELECT t1.team_id, t1.name, t2.team_id, t2.name
                FROM MATCHES m
                JOIN TEAMS t1 ON m.home_team_id = t1.team_id
                JOIN TEAMS t2 ON m.away_team_id = t2.team_id
                WHERE m.match_id = %s
            """, (match_id,))
            match_teams = cur.fetchone()
            cur.close()
            
            if not match_teams:
                print("❌ Could not find teams for this match in database")
                return [], []
            
            team_names = [match_teams[1], match_teams[3]]  # home name, away name
            team_ids_map = {match_teams[1]: match_teams[0], match_teams[3]: match_teams[2]}
            print(f"Found teams from DB: {team_names}")
            
            # FIX: Find lineup tables directly (not divs)
            for team_name in team_names:
                team_id = team_ids_map.get(team_name)
                if not team_id:
                    continue
                
                print(f"✅ Processing lineup for {team_name} (team_id: {team_id})")
                
                # Find ALL tables on the page
                all_tables = soup.find_all('table')
                
                for table in all_tables:
                    # Check if this table has player links
                    player_links_in_table = table.find_all('a', href=re.compile(r'/en/players/'))
                    
                    if len(player_links_in_table) < 5:  # Skip tables with few players
                        continue
                    
                    # Check if table contains our team name in nearby text
                    table_parent = table.find_parent()
                    if table_parent:
                        parent_text = table_parent.get_text()
                        if team_name.lower() not in parent_text.lower():
                            continue
                    
                    # Extract players from this table
                    tbody = table.find('tbody')
                    if not tbody:
                        continue
                    
                    rows = tbody.find_all('tr')
                    
                    for idx, row in enumerate(rows):
                        player_link = row.find('a', href=re.compile(r'/en/players/'))
                        if not player_link:
                            continue
                        
                        player_fbref_id = self.extract_fbref_id(player_link['href'])
                        if not player_fbref_id:
                            continue
                        
                        # Get player_id from database
                        cur = self.conn.cursor()
                        cur.execute("SELECT player_id FROM PLAYERS WHERE fbref_id = %s", (player_fbref_id,))
                        player_result = cur.fetchone()
                        cur.close()
                        
                        if player_result:
                            is_starter = idx < 11
                            
                            lineups.append({
                                'match_id': match_id,
                                'player_id': player_result[0],
                                'team_id': team_id,
                                'is_starter': is_starter,
                                'minutes_played': 90 if is_starter else 0
                            })
                    
                    break  # Found the table for this team, move to next team
            
            # Extract events from scorebox
            scorebox = soup.find('div', class_='scorebox')
            if scorebox:
                score_divs = scorebox.find_all('div', class_=re.compile(r'score', re.I))
                
                for div in score_divs:
                    scorer_elements = div.find_all(['small', 'div'], recursive=True)
                    
                    for elem in scorer_elements:
                        text = elem.text.strip()
                        matches = re.findall(r'([^,]+?)\s+(\d+)(?:\'|\u2032)', text)
                        
                        for player_name, minute in matches:
                            player_name = player_name.strip()
                            minute = int(minute)
                            
                            player_link = elem.find('a', href=re.compile(r'/en/players/'))
                            if not player_link:
                                player_link = elem.find_parent().find('a', href=re.compile(r'/en/players/'))
                            
                            if player_link:
                                player_fbref_id = self.extract_fbref_id(player_link['href'])
                                if not player_fbref_id:
                                    continue
                                
                                # FIX: Match player and verify they're in one of the two teams
                                cur = self.conn.cursor()
                                cur.execute("""
                                    SELECT p.player_id, p.team_id 
                                    FROM PLAYERS p 
                                    WHERE p.fbref_id = %s
                                    AND p.team_id IN (%s, %s)
                                """, (player_fbref_id, team_ids_map[team_names[0]], team_ids_map[team_names[1]]))
                                player_result = cur.fetchone()
                                cur.close()
                                
                                if player_result:
                                    events.append({
                                        'match_id': match_id,
                                        'minute': minute,
                                        'event_type': 'Goal',
                                        'player_id': player_result[0],
                                        'team_id': player_result[1]
                                    })
            
            print(f"✅ Extracted {len(lineups)} lineups, {len(events)} events")
            return lineups, events
            
        except Exception as e:
            print(f"❌ Error scraping match details: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def insert_league(self, name, country, tier, fbref_id):
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO LEAGUES (name, country, tier, fbref_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (fbref_id) DO UPDATE 
                SET name = EXCLUDED.name
                RETURNING league_id
            """, (name, country, tier, fbref_id))
            league_id = cur.fetchone()[0]
            self.conn.commit()
            print(f"✅ Inserted league: {name}")
            return league_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting league: {e}")
            return None
        finally:
            cur.close()
    
    def insert_season(self, year, start_date, end_date, league_id):
        cur = self.conn.cursor()
        try:
            cur.execute("""
                SELECT season_id FROM SEASONS 
                WHERE year = %s AND league_id = %s
            """, (year, league_id))
            existing = cur.fetchone()
            
            if existing:
                season_id = existing[0]
                print(f"ℹ️  Season {year} already exists")
            else:
                cur.execute("""
                    INSERT INTO SEASONS (year, start_date, end_date, league_id)
                    VALUES (%s, %s, %s, %s)
                    RETURNING season_id
                """, (year, start_date, end_date, league_id))
                season_id = cur.fetchone()[0]
                print(f"✅ Inserted season: {year}")
            
            self.conn.commit()
            return season_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting season: {e}")
            return None
        finally:
            cur.close()
    
    def insert_team(self, name, stadium, city, fbref_id):
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO TEAMS (name, stadium_name, city, fbref_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (fbref_id) DO UPDATE 
                SET name = EXCLUDED.name
                RETURNING team_id
            """, (name, stadium, city, fbref_id))
            team_id = cur.fetchone()[0]
            self.conn.commit()
            return team_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting team {name}: {e}")
            return None
        finally:
            cur.close()
    
    def insert_player(self, full_name, nationality, position, team_id, fbref_id, age=None, shirt_number=None):
        """Insert player"""
        cur = self.conn.cursor()
        try:
            pos_map = {
                'GK': 'Goalkeeper', 
                'DF': 'Defender',
                'MF': 'Midfielder', 
                'FW': 'Forward'
            }
            
            if position and pd.notna(position):
                pos_str = str(position).strip()
                if len(pos_str) >= 2:
                    pos_abbr = pos_str[:2].upper()
                    position = pos_map.get(pos_abbr, 'Midfielder')
                else:
                    position = 'Midfielder'
            else:
                position = 'Midfielder'
            
            dob = None
            if age and pd.notna(age) and str(age).strip():
                try:
                    age_int = int(float(str(age).split('-')[0]))
                    current_year = datetime.now().year
                    birth_year = current_year - age_int
                    dob = f"{birth_year}-01-01"
                except:
                    pass
            
            if nationality and pd.notna(nationality):
                nationality = str(nationality).strip()
                nationality = re.sub(r'\s+[A-Z]{3}$', '', nationality)
                nationality = nationality.strip()[:3].upper() if len(nationality) >= 3 else None
            
            if shirt_number and pd.notna(shirt_number):
                try:
                    shirt_number = int(float(shirt_number))
                except:
                    shirt_number = None
            else:
                shirt_number = None
            
            cur.execute("""
                INSERT INTO PLAYERS (full_name, nationality, position, team_id, fbref_id, dob, shirt_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fbref_id) DO UPDATE 
                SET team_id = EXCLUDED.team_id, 
                    shirt_number = EXCLUDED.shirt_number,
                    full_name = EXCLUDED.full_name,
                    position = EXCLUDED.position
                RETURNING player_id
            """, (full_name, nationality, position, team_id, fbref_id, dob, shirt_number))
            player_id = cur.fetchone()[0]
            self.conn.commit()
            return player_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting player {full_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            cur.close()
    
    def insert_match(self, match_date, venue, home_team, away_team, 
                    home_score, away_score, home_xg, away_xg, season_id, fbref_id):
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (home_team,))
            home_id = cur.fetchone()
            cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (away_team,))
            away_id = cur.fetchone()
            
            if not home_id or not away_id:
                return None
            
            cur.execute("""
                INSERT INTO MATCHES 
                (match_date, venue, home_score_final, away_score_final, 
                 home_xg, away_xg, season_id, home_team_id, away_team_id, fbref_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fbref_id) DO UPDATE
                SET home_score_final = EXCLUDED.home_score_final,
                    away_score_final = EXCLUDED.away_score_final
                RETURNING match_id
            """, (match_date, venue, home_score, away_score, 
                  home_xg, away_xg, season_id, home_id[0], away_id[0], fbref_id))
            match_id = cur.fetchone()[0]
            self.conn.commit()
            return match_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting match: {e}")
            return None
        finally:
            cur.close()
    
    def insert_team_season(self, team_id, season_id, standings_row):
        cur = self.conn.cursor()
        try:
            points = int(standings_row.get('Pts', 0)) if 'Pts' in standings_row and pd.notna(standings_row.get('Pts')) else 0
            wins = int(standings_row.get('W', 0)) if 'W' in standings_row and pd.notna(standings_row.get('W')) else 0
            draws = int(standings_row.get('D', 0)) if 'D' in standings_row and pd.notna(standings_row.get('D')) else 0
            losses = int(standings_row.get('L', 0)) if 'L' in standings_row and pd.notna(standings_row.get('L')) else 0
            goals_for = int(standings_row.get('GF', 0)) if 'GF' in standings_row and pd.notna(standings_row.get('GF')) else 0
            goals_against = int(standings_row.get('GA', 0)) if 'GA' in standings_row and pd.notna(standings_row.get('GA')) else 0
            goal_diff = int(standings_row.get('GD', 0)) if 'GD' in standings_row and pd.notna(standings_row.get('GD')) else 0
            
            position = None
            for col in ['Rk', 'Pos', 'Position']:
                if col in standings_row and pd.notna(standings_row.get(col)):
                    try:
                        position = int(standings_row[col])
                        break
                    except:
                        pass
            
            cur.execute("""
                INSERT INTO TEAM_SEASONS 
                (team_id, season_id, points, wins, draws, losses, 
                 goals_for, goals_against, goal_diff, position)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (team_id, season_id) DO UPDATE
                SET points = EXCLUDED.points, wins = EXCLUDED.wins
                RETURNING team_season_id
            """, (team_id, season_id, points, wins, draws, losses,
                  goals_for, goals_against, goal_diff, position))
            
            team_season_id = cur.fetchone()[0]
            self.conn.commit()
            return team_season_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting team_season: {e}")
            return None
        finally:
            cur.close()
    
    def insert_lineup(self, lineup_data):
        """Insert lineup record - FIXED with conflict handling"""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO MATCH_LINEUPS 
                (match_id, player_id, team_id, is_starter, minutes_played)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (match_id, player_id) DO UPDATE
                SET is_starter = EXCLUDED.is_starter,
                    minutes_played = EXCLUDED.minutes_played
                RETURNING lineup_id
            """, (
                lineup_data['match_id'],
                lineup_data['player_id'],
                lineup_data['team_id'],
                lineup_data['is_starter'],
                lineup_data['minutes_played']
            ))
            lineup_id = cur.fetchone()[0]
            self.conn.commit()
            return lineup_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting lineup: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            cur.close()
    
    def insert_event(self, event_data):
        """Insert match event"""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO MATCH_EVENT 
                (match_id, minute, event_type, player_id, team_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING event_id
            """, (
                event_data['match_id'],
                event_data['minute'],
                event_data['event_type'],
                event_data['player_id'],
                event_data['team_id']
            ))
            event_id = cur.fetchone()[0]
            self.conn.commit()
            return event_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting event: {e}")
            return None
        finally:
            cur.close()


def main():
    scraper = FBrefScraper()
    
    try:
        scraper.connect_db()
    except:
        print("❌ Could not connect to database")
        return
    
    try:
        print("\n" + "="*50)
        print("TRUNCATING TABLES")
        print("="*50)
        scraper.truncate_all_tables()
        
        print("\n" + "="*50)
        print("INSERTING LEAGUE")
        print("="*50)
        league_id = scraper.insert_league('Premier League', 'England', 1, '9')
        
        if not league_id:
            return
        
        # MULTIPLE SEASONS - Scrape last 3 seasons
        seasons_config = [
            {
                'year': '2024-25',
                'start': '2024-08-16',
                'end': '2025-05-25',
                'url': 'https://fbref.com/en/comps/9/Premier-League-Stats'
            },
            {
                'year': '2023-24',
                'start': '2023-08-11',
                'end': '2024-05-19',
                'url': 'https://fbref.com/en/comps/9/2023-2024/2023-2024-Premier-League-Stats'
            },
            {
                'year': '2022-23',
                'start': '2022-08-05',
                'end': '2023-05-28',
                'url': 'https://fbref.com/en/comps/9/2022-2023/2022-2023-Premier-League-Stats'
            }
        ]
        
        for season_config in seasons_config:
            print("\n" + "="*70)
            print(f"PROCESSING SEASON: {season_config['year']}")
            print("="*70)
            
            season_id = scraper.insert_season(
                season_config['year'], 
                season_config['start'], 
                season_config['end'], 
                league_id
            )
            
            if not season_id:
                print(f"❌ Failed to create season {season_config['year']}, skipping...")
                continue
            
            league_url = season_config['url']
            
            print("\n" + "="*50)
            print(f"SCRAPING TEAMS - {season_config['year']}")
            print("="*50)
            standings_df = scraper.scrape_league_standings(league_url, season_config['year'])
            
            if standings_df.empty:
                print(f"⚠️  No standings found for {season_config['year']}")
                continue
            
            team_ids = {}
            for _, row in tqdm(standings_df.iterrows(), total=len(standings_df), desc="Inserting teams"):
                team_name = row['Squad']
                fbref_id = row.get('fbref_id', f"team_{team_name.replace(' ', '_').lower()}")
                
                team_id = scraper.insert_team(team_name, None, None, fbref_id)
                if team_id:
                    team_ids[team_name] = {
                        'team_id': team_id,
                        'team_url': row.get('team_url'),
                        'standings_row': row
                    }
            
            print(f"\n✅ Inserted {len(team_ids)} teams for {season_config['year']}")
            
            # Refresh team cache for lineup matching
            scraper.refresh_team_cache()
            
            print("\n" + "="*50)
            print(f"INSERTING TEAM SEASON STATS - {season_config['year']}")
            print("="*50)
            for team_name, team_info in tqdm(team_ids.items(), desc="Team seasons"):
                scraper.insert_team_season(team_info['team_id'], season_id, team_info['standings_row'])
            
            print("\n" + "="*50)
            print(f"SCRAPING PLAYERS - {season_config['year']}")
            print("="*50)
            player_count = 0
            for team_name, team_info in tqdm(list(team_ids.items()), desc="Scraping players"):
                if not team_info.get('team_url'):
                    continue
                
                players_df = scraper.scrape_team_players(team_info['team_url'])
                
                if players_df.empty:
                    print(f"⚠️  No players found for {team_name}")
                    continue
                
                for _, player in players_df.iterrows():
                    try:
                        player_name = player.get('Player')
                        age = player.get('Age')
                        position = player.get('Pos')
                        nationality = player.get('Nation')
                        shirt_number = player.get('#') if '#' in player else player.get('Num')
                        fbref_id = player.get('fbref_id')
                        
                        if not player_name or not fbref_id:
                            continue
                        
                        player_id = scraper.insert_player(
                            player_name, nationality, position, 
                            team_info['team_id'], fbref_id, age, shirt_number
                        )
                        if player_id:
                            player_count += 1
                    except Exception as e:
                        print(f"⚠️  Failed to insert player: {e}")
                        continue
            
            print(f"\n✅ Inserted {player_count} players for {season_config['year']}")
            
            print("\n" + "="*50)
            print(f"SCRAPING MATCHES - {season_config['year']}")
            print("="*50)
            matches_df = scraper.scrape_match_results(league_url, season_config['year'])
            
            if matches_df.empty:
                print(f"⚠️  No matches found for {season_config['year']}")
                continue
            
            match_count = 0
            match_ids_list = []
            
            for _, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Inserting matches"):
                try:
                    score = str(row['Score']).split('–')
                    if len(score) != 2:
                        score = str(row['Score']).split('-')
                    
                    home_score = int(score[0].strip()) if len(score) == 2 else None
                    away_score = int(score[1].strip()) if len(score) == 2 else None
                    
                    home_xg = float(row.get('xG', 0)) if 'xG' in row and pd.notna(row.get('xG')) else None
                    away_xg = float(row.get('xG.1', 0)) if 'xG.1' in row and pd.notna(row.get('xG.1')) else None
                    
                    fbref_id = row.get('fbref_id', f"match_{row['Date']}_{row['Home']}_{row['Away']}")
                    
                    match_id = scraper.insert_match(
                        row['Date'], row.get('Venue', ''),
                        row['Home'], row['Away'],
                        home_score, away_score,
                        home_xg, away_xg,
                        season_id, fbref_id
                    )
                    
                    if match_id:
                        match_count += 1
                        match_ids_list.append({
                            'match_id': match_id,
                            'match_url': row.get('match_url')
                        })
                except Exception as e:
                    print(f"⚠️  Skipped match: {e}")
                    continue
            
            print(f"\n✅ Inserted {match_count} matches for {season_config['year']}")
            
            # Scrape lineups and events for ALL matches (not just first 10)
            print("\n" + "="*50)
            print(f"SCRAPING LINEUPS & EVENTS - {season_config['year']} (ALL MATCHES)")
            print("="*50)
            
            lineup_count = 0
            event_count = 0
            
            # Process ALL matches
            for match_info in tqdm(match_ids_list, desc="Match details"):
                if not match_info.get('match_url'):
                    continue
                
                lineups, events = scraper.scrape_match_lineups_and_events(
                    match_info['match_url'],
                    match_info['match_id']
                )
                
                # Insert lineups
                for lineup in lineups:
                    if scraper.insert_lineup(lineup):
                        lineup_count += 1
                
                # Insert events
                for event in events:
                    if scraper.insert_event(event):
                        event_count += 1
            
            print(f"\n✅ Season {season_config['year']} Summary:")
            print(f"   - Teams: {len(team_ids)}")
            print(f"   - Players: {player_count}")
            print(f"   - Matches: {match_count}")
            print(f"   - Lineups: {lineup_count}")
            print(f"   - Events: {event_count}")
        
        print("\n" + "="*70)
        print("✅ ALL SEASONS SCRAPING COMPLETED!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_db()


if __name__ == '__main__':
    main()