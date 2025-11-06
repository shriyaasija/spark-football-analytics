"""
FBref Data Scraper for SPARK Database
Scrapes Premier League data from fbref.com
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import psycopg2
from datetime import datetime
import os
from dotenv import load_dotenv
from tqdm import tqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# FBref respects rate limits: 3 seconds between requests
RATE_LIMIT = 3

class FBrefScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.base_url = 'https://fbref.com'
        self.conn = None
        
    def connect_db(self):
        """Connect to PostgreSQL database"""
        self.conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            database=os.getenv('POSTGRES_DB', 'spark_db'),
            user=os.getenv('POSTGRES_USER', 'spark_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'spark_password_2024')
        )
        print("✅ Connected to database")
        
    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")
    
    def scrape_league_standings(self, league_url, season_year):
        """
        Scrape league standings and team data
        Example URL: https://fbref.com/en/comps/9/Premier-League-Stats
        """
        print(f"\n📊 Scraping standings from: {league_url}")
        time.sleep(RATE_LIMIT)
        
        response = requests.get(league_url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find standings table
        standings_table = soup.select('table.stats_table')[0]
        df = pd.read_html(str(standings_table))[0]
        
        # Clean dataframe
        df.columns = df.columns.droplevel(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
        df = df[df['Squad'] != 'Squad']  # Remove header rows
        
        print(f"✅ Found {len(df)} teams")
        return df
    
    def scrape_team_players(self, team_url):
        """
        Scrape player data for a specific team
        Example URL: https://fbref.com/en/squads/18bb7c10/Arsenal-Stats
        """
        print(f"\n👥 Scraping players from: {team_url}")
        time.sleep(RATE_LIMIT)
        
        response = requests.get(team_url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find standard stats table
        table = soup.select_one('table[id*="stats_standard"]')
        if not table:
            print("❌ Could not find player table")
            return pd.DataFrame()
        
        df = pd.read_html(str(table))[0]
        
        # Clean multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        # Remove header rows
        df = df[df.iloc[:, 0] != 'Rk']
        
        print(f"✅ Found {len(df)} players")
        return df
    
    def scrape_match_results(self, league_url, season_year):
        """
        Scrape match results with scores and xG
        Example: https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures
        """
        fixtures_url = league_url.replace('Premier-League-Stats', 'schedule/Premier-League-Scores-and-Fixtures')
        print(f"\n⚽ Scraping matches from: {fixtures_url}")
        time.sleep(RATE_LIMIT)
        
        response = requests.get(fixtures_url, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find fixtures table
        table = soup.select('table.stats_table')[0]
        df = pd.read_html(str(table))[0]
        
        # Only keep finished matches
        df = df[df['Score'].notna()]
        df = df[df['Score'] != 'Score']
        
        print(f"✅ Found {len(df)} completed matches")
        return df
    
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
        """Insert season into database"""
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO SEASONS (year, start_date, end_date, league_id)
                VALUES (%s, %s, %s, %s)
                RETURNING season_id
            """, (year, start_date, end_date, league_id))
            season_id = cur.fetchone()[0]
            self.conn.commit()
            print(f"✅ Inserted season: {year} (ID: {season_id})")
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
    
    def insert_player(self, full_name, nationality, position, team_id, fbref_id, age=None):
        """Insert player into database"""
        cur = self.conn.cursor()
        try:
            # Convert position abbreviations
            pos_map = {'GK': 'Goalkeeper', 'DF': 'Defender', 
                      'MF': 'Midfielder', 'FW': 'Forward'}
            position = pos_map.get(position[:2], 'Midfielder')
            
            # Calculate DOB from age if available
            dob = None
            if age and age != '':
                try:
                    current_year = datetime.now().year
                    birth_year = current_year - int(age)
                    dob = f"{birth_year}-01-01"
                except:
                    pass
            
            cur.execute("""
                INSERT INTO PLAYERS (full_name, nationality, position, team_id, fbref_id, dob)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (fbref_id) DO UPDATE 
                SET team_id = EXCLUDED.team_id
                RETURNING player_id
            """, (full_name, nationality, position, team_id, fbref_id, dob))
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
                    home_score, away_score, home_xg, away_xg, season_id):
        """Insert match into database"""
        cur = self.conn.cursor()
        try:
            # Get team IDs
            cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (home_team,))
            home_id = cur.fetchone()
            cur.execute("SELECT team_id FROM TEAMS WHERE name = %s", (away_team,))
            away_id = cur.fetchone()
            
            if not home_id or not away_id:
                return None
            
            cur.execute("""
                INSERT INTO MATCHES 
                (match_date, venue, home_score_final, away_score_final, 
                 home_xg, away_xg, season_id, home_team_id, away_team_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING match_id
            """, (match_date, venue, home_score, away_score, 
                  home_xg, away_xg, season_id, home_id[0], away_id[0]))
            match_id = cur.fetchone()[0]
            self.conn.commit()
            return match_id
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Error inserting match: {e}")
            return None
        finally:
            cur.close()


def main():
    """Main scraping workflow"""
    scraper = FBrefScraper()
    scraper.connect_db()
    
    try:
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
        
        # 2. Insert 2024-25 Season
        print("\n" + "="*50)
        print("STEP 2: INSERT SEASON")
        print("="*50)
        season_id = scraper.insert_season(
            year='2024-25',
            start_date='2024-08-16',
            end_date='2025-05-25',
            league_id=league_id
        )
        
        # 3. Scrape and insert teams
        print("\n" + "="*50)
        print("STEP 3: SCRAPE TEAMS")
        print("="*50)
        league_url = 'https://fbref.com/en/comps/9/Premier-League-Stats'
        standings_df = scraper.scrape_league_standings(league_url, '2024-25')
        
        team_ids = {}
        for _, row in tqdm(standings_df.iterrows(), total=len(standings_df), desc="Inserting teams"):
            team_name = row['Squad']
            team_id = scraper.insert_team(
                name=team_name,
                stadium=None,
                city=None,
                fbref_id=f"team_{team_name.replace(' ', '_').lower()}"
            )
            if team_id:
                team_ids[team_name] = team_id
        
        print(f"\n✅ Inserted {len(team_ids)} teams")
        
        # 4. Scrape matches
        print("\n" + "="*50)
        print("STEP 4: SCRAPE MATCHES")
        print("="*50)
        matches_df = scraper.scrape_match_results(league_url, '2024-25')
        
        match_count = 0
        for _, row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Inserting matches"):
            try:
                # Parse score
                score = str(row['Score']).split('–')
                home_score = int(score[0]) if len(score) == 2 else None
                away_score = int(score[1]) if len(score) == 2 else None
                
                # Parse xG (if available)
                home_xg = float(row.get('xG', 0)) if 'xG' in row else None
                away_xg = float(row.get('xG.1', 0)) if 'xG.1' in row else None
                
                match_id = scraper.insert_match(
                    match_date=row['Date'],
                    venue=row.get('Venue', ''),
                    home_team=row['Home'],
                    away_team=row['Away'],
                    home_score=home_score,
                    away_score=away_score,
                    home_xg=home_xg,
                    away_xg=away_xg,
                    season_id=season_id
                )
                if match_id:
                    match_count += 1
            except Exception as e:
                continue
        
        print(f"\n✅ Inserted {match_count} matches")
        
        print("\n" + "="*50)
        print("✅ SCRAPING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        scraper.close_db()


if __name__ == '__main__':
    main()