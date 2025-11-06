# """
# FBref Data Scraper with Cloudflare Bypass
# Uses cloudscraper to bypass Cloudflare protection
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

# load_dotenv()

# # Respect rate limits: 3-5 seconds between requests
# RATE_LIMIT_MIN = 3
# RATE_LIMIT_MAX = 5

# class FBrefScraper:
#     def __init__(self):
#         # Create cloudscraper instance
#         self.scraper = cloudscraper.create_scraper(
#             browser={
#                 'browser': 'chrome',
#                 'platform': 'windows',
#                 'mobile': False
#             },
#             delay=10  # Delay to solve Cloudflare challenge
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
#         """Close database connection"""
#         if self.conn:
#             self.conn.close()
#             print("✅ Database connection closed")
    
#     def random_delay(self):
#         """Add random delay between requests"""
#         delay = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
#         time.sleep(delay)
    
#     def scrape_league_standings(self, league_url, season_year):
#         """
#         Scrape league standings and team data
#         Example URL: https://fbref.com/en/comps/9/Premier-League-Stats
#         """
#         print(f"\n📊 Scraping standings from: {league_url}")
#         self.random_delay()
        
#         try:
#             # Use cloudscraper instead of requests
#             response = self.scraper.get(league_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             print(f"✅ Successfully fetched page (Status: {response.status_code})")
            
#             # Find ALL tables and try to get the standings
#             tables = soup.find_all('table')
#             print(f"Found {len(tables)} tables on page")
            
#             standings_df = None
#             for i, table in enumerate(tables):
#                 try:
#                     df = pd.read_html(str(table))[0]
#                     # Check if this looks like a standings table
#                     if 'Squad' in str(df.columns) or 'Team' in str(df.columns):
#                         standings_df = df
#                         print(f"✅ Found standings table at index {i}")
#                         break
#                 except Exception as e:
#                     continue
            
#             if standings_df is None:
#                 print("❌ Could not find standings table")
#                 return pd.DataFrame()
            
#             # Clean dataframe
#             if isinstance(standings_df.columns, pd.MultiIndex):
#                 standings_df.columns = standings_df.columns.droplevel(0)
            
#             # Find the Squad column (might be 'Squad' or 'Team')
#             squad_col = None
#             for col in standings_df.columns:
#                 if 'Squad' in str(col) or 'Team' in str(col):
#                     squad_col = col
#                     break
            
#             if squad_col:
#                 standings_df = standings_df[standings_df[squad_col] != squad_col]
#                 standings_df = standings_df.rename(columns={squad_col: 'Squad'})
            
#             print(f"✅ Found {len(standings_df)} teams")
#             print(f"Teams: {standings_df['Squad'].tolist()[:5]}...")
#             return standings_df
            
#         except cloudscraper.exceptions.CloudflareChallengeError as e:
#             print(f"❌ Cloudflare challenge failed: {e}")
#             print("💡 Try increasing the delay parameter or using Selenium")
#             return pd.DataFrame()
#         except Exception as e:
#             print(f"❌ Error scraping standings: {e}")
#             return pd.DataFrame()
    
#     def scrape_team_players(self, team_url):
#         """
#         Scrape player data for a specific team
#         Example URL: https://fbref.com/en/squads/18bb7c10/Arsenal-Stats
#         """
#         print(f"\n👥 Scraping players from: {team_url}")
#         self.random_delay()
        
#         try:
#             response = self.scraper.get(team_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # Find standard stats table
#             table = soup.select_one('table[id*="stats_standard"]')
#             if not table:
#                 print("❌ Could not find player table")
#                 return pd.DataFrame()
            
#             df = pd.read_html(str(table))[0]
            
#             # Clean multi-index columns
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
#             # Remove header rows
#             df = df[df.iloc[:, 0] != 'Rk']
            
#             print(f"✅ Found {len(df)} players")
#             return df
#         except Exception as e:
#             print(f"❌ Error scraping players: {e}")
#             return pd.DataFrame()
    
#     def scrape_match_results(self, league_url, season_year):
#         """
#         Scrape match results with scores and xG
#         Example: https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures
#         """
#         fixtures_url = league_url.replace('Premier-League-Stats', 'schedule/Premier-League-Scores-and-Fixtures')
#         print(f"\n⚽ Scraping matches from: {fixtures_url}")
#         self.random_delay()
        
#         try:
#             response = self.scraper.get(fixtures_url)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # Find fixtures table
#             tables = soup.find_all('table')
#             if not tables:
#                 print("❌ No tables found")
#                 return pd.DataFrame()
            
#             df = pd.read_html(str(tables[0]))[0]
            
#             # Clean multi-index if present
#             if isinstance(df.columns, pd.MultiIndex):
#                 df.columns = df.columns.droplevel(0)
            
#             # Only keep finished matches (those with scores)
#             if 'Score' in df.columns:
#                 df = df[df['Score'].notna()]
#                 df = df[df['Score'] != 'Score']
            
#             print(f"✅ Found {len(df)} completed matches")
#             return df
#         except Exception as e:
#             print(f"❌ Error scraping matches: {e}")
#             return pd.DataFrame()
    
#     def insert_league(self, name, country, tier, fbref_id):
#         """Insert league into database"""
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
#             print(f"✅ Inserted league: {name} (ID: {league_id})")
#             return league_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting league: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_season(self, year, start_date, end_date, league_id):
#         """Insert season into database"""
#         cur = self.conn.cursor()
#         try:
#             cur.execute("""
#                 INSERT INTO SEASONS (year, start_date, end_date, league_id)
#                 VALUES (%s, %s, %s, %s)
#                 RETURNING season_id
#             """, (year, start_date, end_date, league_id))
#             season_id = cur.fetchone()[0]
#             self.conn.commit()
#             print(f"✅ Inserted season: {year} (ID: {season_id})")
#             return season_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting season: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_team(self, name, stadium, city, fbref_id):
#         """Insert team into database"""
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
    
#     def insert_player(self, full_name, nationality, position, team_id, fbref_id, age=None):
#         """Insert player into database"""
#         cur = self.conn.cursor()
#         try:
#             # Convert position abbreviations
#             pos_map = {'GK': 'Goalkeeper', 'DF': 'Defender', 
#                       'MF': 'Midfielder', 'FW': 'Forward'}
#             position = pos_map.get(position[:2], 'Midfielder')
            
#             # Calculate DOB from age if available
#             dob = None
#             if age and age != '':
#                 try:
#                     current_year = datetime.now().year
#                     birth_year = current_year - int(age)
#                     dob = f"{birth_year}-01-01"
#                 except:
#                     pass
            
#             cur.execute("""
#                 INSERT INTO PLAYERS (full_name, nationality, position, team_id, fbref_id, dob)
#                 VALUES (%s, %s, %s, %s, %s, %s)
#                 ON CONFLICT (fbref_id) DO UPDATE 
#                 SET team_id = EXCLUDED.team_id
#                 RETURNING player_id
#             """, (full_name, nationality, position, team_id, fbref_id, dob))
#             player_id = cur.fetchone()[0]
#             self.conn.commit()
#             return player_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting player {full_name}: {e}")
#             return None
#         finally:
#             cur.close()
    
#     def insert_match(self, match_date, venue, home_team, away_team, 
#                     home_score, away_score, home_xg, away_xg, season_id):
#         """Insert match into database"""
#         cur = self.conn.cursor()
#         try:
#             # Get team IDs
#             cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (home_team,))
#             home_id = cur.fetchone()
#             cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (away_team,))
#             away_id = cur.fetchone()
            
#             if not home_id or not away_id:
#                 print(f"⚠️  Teams not found: {home_team} vs {away_team}")
#                 return None
            
#             cur.execute("""
#                 INSERT INTO MATCHES 
#                 (match_date, venue, home_score_final, away_score_final, 
#                  home_xg, away_xg, season_id, home_team_id, away_team_id)
#                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
#                 RETURNING match_id
#             """, (match_date, venue, home_score, away_score, 
#                   home_xg, away_xg, season_id, home_id[0], away_id[0]))
#             match_id = cur.fetchone()[0]
#             self.conn.commit()
#             return match_id
#         except Exception as e:
#             self.conn.rollback()
#             print(f"❌ Error inserting match: {e}")
#             return None
#         finally:
#             cur.close()


# def main():
#     """Main scraping workflow"""
#     scraper = FBrefScraper()
    
#     try:
#         scraper.connect_db()
#     except:
#         print("❌ Could not connect to database. Exiting.")
#         return
    
#     try:
#         # 1. Insert Premier League
#         print("\n" + "="*50)
#         print("STEP 1: INSERT LEAGUE")
#         print("="*50)
#         league_id = scraper.insert_league(
#             name='Premier League',
#             country='England',
#             tier=1,
#             fbref_id='9'
#         )
        
#         if not league_id:
#             print("❌ Failed to insert league")
#             return
        
#         # 2. Insert 2024-25 Season
#         print("\n" + "="*50)
#         print("STEP 2: INSERT SEASON")
#         print("="*50)
#         season_id = scraper.insert_season(
#             year='2024-25',
#             start_date='2024-08-16',
#             end_date='2025-05-25',
#             league_id=league_id
#         )
        
#         if not season_id:
#             print("❌ Failed to insert season")
#             return
        
#         # 3. Scrape and insert teams
#         print("\n" + "="*50)
#         print("STEP 3: SCRAPE TEAMS")
#         print("="*50)
#         league_url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
#         standings_df = scraper.scrape_league_standings(league_url, '2024-25')
        
#         if standings_df.empty:
#             print("❌ No teams found. Stopping.")
#             return
        
#         team_ids = {}
#         for _, row in tqdm(standings_df.iterrows(), total=len(standings_df), desc="Inserting teams"):
#             team_name = row['Squad']
#             team_id = scraper.insert_team(
#                 name=team_name,
#                 stadium=None,
#                 city=None,
#                 fbref_id=f"team_{team_name.replace(' ', '_').lower()}"
#             )
#             if team_id:
#                 team_ids[team_name] = team_id
        
#         print(f"\n✅ Inserted {len(team_ids)} teams")
        
#         # 4. Scrape matches
#         print("\n" + "="*50)
#         print("STEP 4: SCRAPE MATCHES")
#         print("="*50)
#         matches_df = scraper.scrape_match_results(league_url, '2024-25')
        
#         if matches_df.empty:
#             print("⚠️  No matches found")
#         else:
#             match_count = 0
#             for _, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Inserting matches"):
#                 try:
#                     # Parse score
#                     score = str(row['Score']).split('–')
#                     if len(score) != 2:
#                         score = str(row['Score']).split('-')
                    
#                     home_score = int(score[0].strip()) if len(score) == 2 else None
#                     away_score = int(score[1].strip()) if len(score) == 2 else None
                    
#                     # Parse xG (if available)
#                     home_xg = float(row.get('xG', 0)) if 'xG' in row and pd.notna(row.get('xG')) else None
#                     away_xg = float(row.get('xG.1', 0)) if 'xG.1' in row and pd.notna(row.get('xG.1')) else None
                    
#                     match_id = scraper.insert_match(
#                         match_date=row['Date'],
#                         venue=row.get('Venue', ''),
#                         home_team=row['Home'],
#                         away_team=row['Away'],
#                         home_score=home_score,
#                         away_score=away_score,
#                         home_xg=home_xg,
#                         away_xg=away_xg,
#                         season_id=season_id
#                     )
#                     if match_id:
#                         match_count += 1
#                 except Exception as e:
#                     print(f"⚠️  Skipped match: {e}")
#                     continue
            
#             print(f"\n✅ Inserted {match_count} matches")
        
#         print("\n" + "="*50)
#         print("✅ SCRAPING COMPLETED SUCCESSFULLY!")
#         print("="*50)
        
#     except Exception as e:
#         print(f"\n❌ Fatal error: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         scraper.close_db()


# if __name__ == '__main__':
#     main()

"""
FBref Data Scraper with Cloudflare Bypass - Complete Version
Scrapes all tables with proper fbref_id extraction and data truncation
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

# Respect rate limits: 3-5 seconds between requests
RATE_LIMIT_MIN = 3
RATE_LIMIT_MAX = 5

class FBrefScraper:
    def __init__(self):
        # Create cloudscraper instance
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
        """Close database connection"""
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
    
    def random_delay(self):
        """Add random delay between requests"""
        delay = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
        time.sleep(delay)
    
    def extract_fbref_id(self, url):
        """Extract FBref ID from URL"""
        # FBref URLs format: /en/squads/18bb7c10/Arsenal-Stats
        # Extract: 18bb7c10
        match = re.search(r'/([a-f0-9]{8})/', url)
        if match:
            return match.group(1)
        return None
    
    def scrape_league_standings(self, league_url, season_year):
        """
        Scrape league standings with team URLs and FBref IDs
        Returns: DataFrame with Squad, URL, fbref_id columns
        """
        print(f"\n📊 Scraping standings from: {league_url}")
        self.random_delay()
        
        try:
            response = self.scraper.get(league_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            print(f"✅ Successfully fetched page (Status: {response.status_code})")
            
            # Find standings table
            table = soup.find('table', {'id': re.compile(r'.*standings.*', re.I)})
            if not table:
                # Fallback: find first table with Squad column
                tables = soup.find_all('table')
                for t in tables:
                    if 'Squad' in str(t) or 'Team' in str(t):
                        table = t
                        break
            
            if not table:
                print("❌ Could not find standings table")
                return pd.DataFrame()
            
            # Parse table
            df = pd.read_html(str(table))[0]
            
            # Clean multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            # Find Squad column
            squad_col = None
            for col in df.columns:
                if 'Squad' in str(col) or 'Team' in str(col):
                    squad_col = col
                    break
            
            if not squad_col:
                print("❌ Could not find Squad column")
                return pd.DataFrame()
            
            # Remove duplicate header rows
            df = df[df[squad_col] != squad_col]
            df = df.rename(columns={squad_col: 'Squad'})
            
            # Extract team URLs and FBref IDs
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
            
            # Merge with standings data
            team_df = pd.DataFrame(team_data)
            df = df.merge(team_df, on='Squad', how='left')
            
            print(f"✅ Found {len(df)} teams with URLs")
            print(f"Teams: {df['Squad'].tolist()[:5]}...")
            return df
            
        except cloudscraper.exceptions.CloudflareChallengeError as e:
            print(f"❌ Cloudflare challenge failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error scraping standings: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def scrape_team_players(self, team_url):
        """
        Scrape player data for a specific team
        Returns: DataFrame with player info including fbref_id
        """
        print(f"\n👥 Scraping players from: {team_url}")
        self.random_delay()
        
        try:
            response = self.scraper.get(team_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find standard stats table
            table = soup.find('table', {'id': re.compile(r'.*stats_standard.*', re.I)})
            if not table:
                print("❌ Could not find player table")
                return pd.DataFrame()
            
            df = pd.read_html(str(table))[0]
            
            # Clean multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(str(col)).strip() for col in df.columns.values]
            
            # Remove header rows
            df = df[df.iloc[:, 0] != 'Rk']
            
            # Extract player URLs and FBref IDs
            player_links = table.find_all('a', href=re.compile(r'/en/players/'))
            
            player_data = []
            for link in player_links:
                player_name = link.text.strip()
                player_url = self.base_url + link['href']
                fbref_id = self.extract_fbref_id(link['href'])
                player_data.append({
                    'player_name': player_name,
                    'player_url': player_url,
                    'fbref_id': fbref_id if fbref_id else player_url
                })
            
            player_df = pd.DataFrame(player_data)
            
            # Merge with stats data
            # Find the name column (usually 'Player' or contains 'Player')
            name_col = None
            for col in df.columns:
                if 'Player' in col:
                    name_col = col
                    break
            
            if name_col and not player_df.empty:
                df = df.merge(player_df, left_on=name_col, right_on='player_name', how='left')
            
            print(f"✅ Found {len(df)} players")
            return df
        except Exception as e:
            print(f"❌ Error scraping players: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def scrape_match_results(self, league_url, season_year):
        """
        Scrape match results with scores, xG, and match URLs
        Returns: DataFrame with match info including fbref_id
        """
        fixtures_url = league_url.replace('Premier-League-Stats', 'schedule/Premier-League-Scores-and-Fixtures')
        print(f"\n⚽ Scraping matches from: {fixtures_url}")
        self.random_delay()
        
        try:
            response = self.scraper.get(fixtures_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find fixtures table
            table = soup.find('table', {'id': re.compile(r'.*sched.*', re.I)})
            if not table:
                tables = soup.find_all('table')
                if tables:
                    table = tables[0]
            
            if not table:
                print("❌ No tables found")
                return pd.DataFrame()
            
            df = pd.read_html(str(table))[0]
            
            # Clean multi-index if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
            
            # Only keep finished matches (those with scores)
            if 'Score' in df.columns:
                df = df[df['Score'].notna()]
                df = df[df['Score'] != 'Score']
            
            # Extract match URLs and FBref IDs
            match_links = table.find_all('a', href=re.compile(r'/en/matches/'))
            
            match_data = []
            for link in match_links:
                match_url = self.base_url + link['href']
                fbref_id = self.extract_fbref_id(link['href'])
                match_data.append({
                    'match_url': match_url,
                    'fbref_id': fbref_id if fbref_id else match_url
                })
            
            if match_data:
                match_df = pd.DataFrame(match_data)
                # Add match URLs to first N rows
                for i, row in enumerate(match_data[:len(df)]):
                    if i < len(df):
                        df.loc[df.index[i], 'match_url'] = row['match_url']
                        df.loc[df.index[i], 'fbref_id'] = row['fbref_id']
            
            print(f"✅ Found {len(df)} completed matches")
            return df
        except Exception as e:
            print(f"❌ Error scraping matches: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def insert_league(self, name, country, tier, fbref_id):
        """Insert league into database"""
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
            print(f"✅ Inserted league: {name} (ID: {league_id})")
            return league_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting league: {e}")
            return None
        finally:
            cur.close()
    
    def insert_season(self, year, start_date, end_date, league_id):
        """Insert season into database (with duplicate check)"""
        cur = self.conn.cursor()
        try:
            # Check if season already exists
            cur.execute("""
                SELECT season_id FROM SEASONS 
                WHERE year = %s AND league_id = %s
            """, (year, league_id))
            existing = cur.fetchone()
            
            if existing:
                season_id = existing[0]
                print(f"ℹ️  Season {year} already exists (ID: {season_id})")
            else:
                cur.execute("""
                    INSERT INTO SEASONS (year, start_date, end_date, league_id)
                    VALUES (%s, %s, %s, %s)
                    RETURNING season_id
                """, (year, start_date, end_date, league_id))
                season_id = cur.fetchone()[0]
                print(f"✅ Inserted season: {year} (ID: {season_id})")
            
            self.conn.commit()
            return season_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting season: {e}")
            return None
        finally:
            cur.close()
    
    def insert_team(self, name, stadium, city, fbref_id):
        """Insert team into database"""
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
        """Insert player into database"""
        cur = self.conn.cursor()
        try:
            # Convert position abbreviations
            pos_map = {
                'GK': 'Goalkeeper', 
                'DF': 'Defender',
                'MF': 'Midfielder', 
                'FW': 'Forward'
            }
            
            # Handle various position formats
            if position and len(str(position)) >= 2:
                pos_abbr = str(position)[:2].upper()
                position = pos_map.get(pos_abbr, 'Midfielder')
            else:
                position = 'Midfielder'
            
            # Calculate DOB from age if available
            dob = None
            if age and str(age).strip() and str(age) != 'nan':
                try:
                    current_year = datetime.now().year
                    birth_year = current_year - int(float(age))
                    dob = f"{birth_year}-01-01"
                except:
                    pass
            
            cur.execute("""
                INSERT INTO PLAYERS (full_name, nationality, position, team_id, fbref_id, dob, shirt_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (fbref_id) DO UPDATE 
                SET team_id = EXCLUDED.team_id, shirt_number = EXCLUDED.shirt_number
                RETURNING player_id
            """, (full_name, nationality, position, team_id, fbref_id, dob, shirt_number))
            player_id = cur.fetchone()[0]
            self.conn.commit()
            return player_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting player {full_name}: {e}")
            return None
        finally:
            cur.close()
    
    def insert_match(self, match_date, venue, home_team, away_team, 
                    home_score, away_score, home_xg, away_xg, season_id, fbref_id):
        """Insert match into database"""
        cur = self.conn.cursor()
        try:
            # Get team IDs
            cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (home_team,))
            home_id = cur.fetchone()
            cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (away_team,))
            away_id = cur.fetchone()
            
            if not home_id or not away_id:
                print(f"⚠️  Teams not found: {home_team} vs {away_team}")
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
        """Insert team season stats"""
        cur = self.conn.cursor()
        try:
            # Extract stats from standings row
            points = int(standings_row.get('Pts', 0)) if 'Pts' in standings_row else 0
            wins = int(standings_row.get('W', 0)) if 'W' in standings_row else 0
            draws = int(standings_row.get('D', 0)) if 'D' in standings_row else 0
            losses = int(standings_row.get('L', 0)) if 'L' in standings_row else 0
            goals_for = int(standings_row.get('GF', 0)) if 'GF' in standings_row else 0
            goals_against = int(standings_row.get('GA', 0)) if 'GA' in standings_row else 0
            goal_diff = int(standings_row.get('GD', 0)) if 'GD' in standings_row else 0
            
            # Position might be in 'Rk' column
            position = None
            for col in ['Rk', 'Pos', 'Position']:
                if col in standings_row:
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
                SET points = EXCLUDED.points, wins = EXCLUDED.wins,
                    draws = EXCLUDED.draws, losses = EXCLUDED.losses
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


def main():
    """Main scraping workflow"""
    scraper = FBrefScraper()
    
    try:
        scraper.connect_db()
    except:
        print("❌ Could not connect to database. Exiting.")
        return
    
    try:
        # TRUNCATE ALL TABLES FIRST
        print("\n" + "="*50)
        print("STEP 0: TRUNCATE TABLES")
        print("="*50)
        scraper.truncate_all_tables()
        
        # 1. Insert Premier League
        print("\n" + "="*50)
        print("STEP 1: INSERT LEAGUE")
        print("="*50)
        league_id = scraper.insert_league(
            name='Premier League',
            country='England',
            tier=1,
            fbref_id='9'
        )
        
        if not league_id:
            print("❌ Failed to insert league")
            return
        
        # 2. Insert 2024-25 Season (ONLY ONCE)
        print("\n" + "="*50)
        print("STEP 2: INSERT SEASON")
        print("="*50)
        season_id = scraper.insert_season(
            year='2024-25',
            start_date='2024-08-16',
            end_date='2025-05-25',
            league_id=league_id
        )
        
        if not season_id:
            print("❌ Failed to insert season")
            return
        
        # 3. Scrape and insert teams
        print("\n" + "="*50)
        print("STEP 3: SCRAPE TEAMS")
        print("="*50)
        league_url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
        standings_df = scraper.scrape_league_standings(league_url, '2024-25')
        
        if standings_df.empty:
            print("❌ No teams found. Stopping.")
            return
        
        team_ids = {}
        for _, row in tqdm(standings_df.iterrows(), total=len(standings_df), desc="Inserting teams"):
            team_name = row['Squad']
            fbref_id = row.get('fbref_id', f"team_{team_name.replace(' ', '_').lower()}")
            
            team_id = scraper.insert_team(
                name=team_name,
                stadium=None,
                city=None,
                fbref_id=fbref_id
            )
            if team_id:
                team_ids[team_name] = {
                    'team_id': team_id,
                    'team_url': row.get('team_url'),
                    'standings_row': row
                }
        
        print(f"\n✅ Inserted {len(team_ids)} teams")
        
        # 4. Insert team season stats
        print("\n" + "="*50)
        print("STEP 4: INSERT TEAM SEASON STATS")
        print("="*50)
        for team_name, team_info in tqdm(team_ids.items(), desc="Inserting team season stats"):
            scraper.insert_team_season(
                team_id=team_info['team_id'],
                season_id=season_id,
                standings_row=team_info['standings_row']
            )
        
        # 5. Scrape and insert players
        print("\n" + "="*50)
        print("STEP 5: SCRAPE PLAYERS")
        print("="*50)
        player_count = 0
        for team_name, team_info in tqdm(list(team_ids.items())[:5], desc="Scraping players"):  # Limit to 5 teams for testing
            if not team_info.get('team_url'):
                continue
            
            players_df = scraper.scrape_team_players(team_info['team_url'])
            
            if players_df.empty:
                continue
            
            # Find relevant columns
            name_col = next((col for col in players_df.columns if 'Player' in col), None)
            age_col = next((col for col in players_df.columns if 'Age' in col), None)
            pos_col = next((col for col in players_df.columns if 'Pos' in col), None)
            nation_col = next((col for col in players_df.columns if 'Nation' in col), None)
            number_col = next((col for col in players_df.columns if '#' in col or 'Num' in col), None)
            
            for _, player in players_df.iterrows():
                if name_col and name_col in player:
                    player_name = player[name_col]
                    age = player.get(age_col) if age_col else None
                    position = player.get(pos_col) if pos_col else 'MF'
                    nationality = player.get(nation_col) if nation_col else None
                    shirt_number = player.get(number_col) if number_col else None
                    fbref_id = player.get('fbref_id', f"player_{player_name.replace(' ', '_').lower()}")
                    
                    player_id = scraper.insert_player(
                        full_name=player_name,
                        nationality=nationality,
                        position=position,
                        team_id=team_info['team_id'],
                        fbref_id=fbref_id,
                        age=age,
                        shirt_number=shirt_number
                    )
                    if player_id:
                        player_count += 1
        
        print(f"\n✅ Inserted {player_count} players")
        
        # 6. Scrape matches
        print("\n" + "="*50)
        print("STEP 6: SCRAPE MATCHES")
        print("="*50)
        matches_df = scraper.scrape_match_results(league_url, '2024-25')
        
        if matches_df.empty:
            print("⚠️  No matches found")
        else:
            match_count = 0
            for _, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Inserting matches"):
                try:
                    # Parse score
                    score = str(row['Score']).split('–')
                    if len(score) != 2:
                        score = str(row['Score']).split('-')
                    
                    home_score = int(score[0].strip()) if len(score) == 2 else None
                    away_score = int(score[1].strip()) if len(score) == 2 else None
                    
                    # Parse xG (if available)
                    home_xg = float(row.get('xG', 0)) if 'xG' in row and pd.notna(row.get('xG')) else None
                    away_xg = float(row.get('xG.1', 0)) if 'xG.1' in row and pd.notna(row.get('xG.1')) else None
                    
                    fbref_id = row.get('fbref_id', f"match_{row['Date']}_{row['Home']}_{row['Away']}")
                    
                    match_id = scraper.insert_match(
                        match_date=row['Date'],
                        venue=row.get('Venue', ''),
                        home_team=row['Home'],
                        away_team=row['Away'],
                        home_score=home_score,
                        away_score=away_score,
                        home_xg=home_xg,
                        away_xg=away_xg,
                        season_id=season_id,
                        fbref_id=fbref_id
                    )
                    if match_id:
                        match_count += 1
                except Exception as e:
                    print(f"⚠️  Skipped match: {e}")
                    continue
            
            print(f"\n✅ Inserted {match_count} matches")
        
        print("\n" + "="*50)
        print("✅ SCRAPING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_db()


if __name__ == '__main__':
    main()