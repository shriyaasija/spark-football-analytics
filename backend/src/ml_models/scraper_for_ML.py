"""
Enhanced FBref Scraper - ML Feature Extraction
Extracts comprehensive match statistics for ML model training
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

RATE_LIMIT_MIN = 5
RATE_LIMIT_MAX = 8

class EnhancedFBrefScraper:
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
    
    def random_delay(self):
        delay = random.uniform(RATE_LIMIT_MIN, RATE_LIMIT_MAX)
        time.sleep(delay)
    
    def extract_fbref_id(self, url):
        """Extract FBref ID from URL"""
        match = re.search(r'/([a-f0-9]{8})/', url)
        if match:
            return match.group(1)
        return None
    
    def scrape_detailed_match_stats(self, match_url):
        """
        Scrape detailed match statistics for ML features
        Returns: Dictionary with comprehensive match stats
        """
        print(f"\n📊 Scraping detailed stats: {match_url}")
        self.random_delay()
        
        stats = {
            'match_url': match_url,
            'home_team': None,
            'away_team': None,
            'home_score': None,
            'away_score': None,
            'home_xg': None,
            'away_xg': None,
            'home_possession': None,
            'away_possession': None,
            'home_shots': None,
            'away_shots': None,
            'home_shots_on_target': None,
            'away_shots_on_target': None,
            'home_corners': None,
            'away_corners': None,
            'home_fouls': None,
            'away_fouls': None,
            'home_yellow_cards': None,
            'away_yellow_cards': None,
            'home_red_cards': None,
            'away_red_cards': None,
            'home_passes': None,
            'away_passes': None,
            'home_pass_accuracy': None,
            'away_pass_accuracy': None,
            'home_tackles': None,
            'away_tackles': None,
            'home_interceptions': None,
            'away_interceptions': None,
        }
        
        try:
            response = self.scraper.get(match_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract team names and score from scorebox
            scorebox = soup.find('div', class_='scorebox')
            if scorebox:
                teams = scorebox.find_all('strong')
                if len(teams) >= 2:
                    stats['home_team'] = teams[0].text.strip()
                    stats['away_team'] = teams[1].text.strip()
                
                scores = scorebox.find_all('div', class_='score')
                if len(scores) >= 2:
                    try:
                        stats['home_score'] = int(scores[0].text.strip())
                        stats['away_score'] = int(scores[1].text.strip())
                    except:
                        pass
            
            # Extract team stats table
            team_stats_table = soup.find('table', id=re.compile(r'team_stats'))
            if team_stats_table:
                rows = team_stats_table.find_all('tr')
                for row in rows:
                    th = row.find('th')
                    tds = row.find_all('td')
                    
                    if not th or len(tds) < 2:
                        continue
                    
                    stat_name = th.text.strip().lower()
                    home_val = tds[0].text.strip()
                    away_val = tds[1].text.strip()
                    
                    # Map stats to our dictionary
                    if 'possession' in stat_name:
                        stats['home_possession'] = self._parse_percentage(home_val)
                        stats['away_possession'] = self._parse_percentage(away_val)
                    elif 'shots on target' in stat_name:
                        stats['home_shots_on_target'] = self._parse_number(home_val)
                        stats['away_shots_on_target'] = self._parse_number(away_val)
                    elif stat_name == 'shots':
                        stats['home_shots'] = self._parse_number(home_val)
                        stats['away_shots'] = self._parse_number(away_val)
                    elif 'corners' in stat_name:
                        stats['home_corners'] = self._parse_number(home_val)
                        stats['away_corners'] = self._parse_number(away_val)
                    elif 'fouls' in stat_name:
                        stats['home_fouls'] = self._parse_number(home_val)
                        stats['away_fouls'] = self._parse_number(away_val)
                    elif 'yellow cards' in stat_name:
                        stats['home_yellow_cards'] = self._parse_number(home_val)
                        stats['away_yellow_cards'] = self._parse_number(away_val)
                    elif 'red cards' in stat_name:
                        stats['home_red_cards'] = self._parse_number(home_val)
                        stats['away_red_cards'] = self._parse_number(away_val)
                    elif 'passes' in stat_name and 'accuracy' not in stat_name:
                        stats['home_passes'] = self._parse_number(home_val)
                        stats['away_passes'] = self._parse_number(away_val)
                    elif 'pass accuracy' in stat_name:
                        stats['home_pass_accuracy'] = self._parse_percentage(home_val)
                        stats['away_pass_accuracy'] = self._parse_percentage(away_val)
                    elif 'tackles' in stat_name:
                        stats['home_tackles'] = self._parse_number(home_val)
                        stats['away_tackles'] = self._parse_number(away_val)
                    elif 'interceptions' in stat_name:
                        stats['home_interceptions'] = self._parse_number(home_val)
                        stats['away_interceptions'] = self._parse_number(away_val)
            
            # Extract xG from tables
            for table in soup.find_all('table'):
                if 'summary' in table.get('id', ''):
                    df = pd.read_html(str(table))[0]
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
                    
                    # Sum xG from player stats
                    if 'xG' in df.columns:
                        try:
                            xg_sum = pd.to_numeric(df['xG'], errors='coerce').sum()
                            # Assign to home or away based on table context
                            table_id = table.get('id', '')
                            if stats['home_team'] and stats['home_team'].lower().replace(' ', '_') in table_id.lower():
                                stats['home_xg'] = round(xg_sum, 2)
                            elif stats['away_team'] and stats['away_team'].lower().replace(' ', '_') in table_id.lower():
                                stats['away_xg'] = round(xg_sum, 2)
                        except:
                            pass
            
            print(f"✅ Extracted detailed stats")
            return stats
            
        except Exception as e:
            print(f"❌ Error scraping detailed stats: {e}")
            import traceback
            traceback.print_exc()
            return stats
    
    def _parse_number(self, value):
        """Parse number from string"""
        try:
            return int(value.replace(',', ''))
        except:
            return None
    
    def _parse_percentage(self, value):
        """Parse percentage from string"""
        try:
            return float(value.replace('%', ''))
        except:
            return None
    
    def scrape_player_season_stats(self, team_url):
        """
        Scrape detailed player statistics for ML features
        Returns: DataFrame with player stats
        """
        print(f"\n👤 Scraping player stats: {team_url}")
        self.random_delay()
        
        try:
            response = self.scraper.get(team_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find standard stats table
            table = soup.find('table', {'id': re.compile(r'.*stats_standard.*', re.I)})
            if not table:
                return pd.DataFrame()
            
            df = pd.read_html(str(table))[0]
            
            # Flatten multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[1] if col[1] else col[0] for col in df.columns.values]
            
            # Remove header rows
            df = df[df['Player'] != 'Player']
            
            # Get player links
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
            
            # Select relevant columns for ML
            relevant_cols = ['Player', 'fbref_id', 'Pos', 'Age', 'MP', 'Starts', 'Min', 
                           'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR']
            
            # Add xG, xAG if available
            if 'xG' in df.columns:
                relevant_cols.append('xG')
            if 'xAG' in df.columns:
                relevant_cols.append('xAG')
            if 'npxG' in df.columns:
                relevant_cols.append('npxG')
            
            # Filter to available columns
            available_cols = [col for col in relevant_cols if col in df.columns]
            df = df[available_cols]
            
            print(f"✅ Extracted stats for {len(df)} players")
            return df
            
        except Exception as e:
            print(f"❌ Error scraping player stats: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def export_ml_training_data(self, output_dir='ml_data'):
        """
        Export comprehensive training data for ML models
        Creates multiple CSV files for different prediction tasks
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("EXPORTING ML TRAINING DATA")
        print("="*60)
        
        cur = self.conn.cursor()
        
        # 1. Match outcome training data
        print("\n📊 Exporting match outcome data...")
        query = """
        SELECT 
            m.match_id,
            m.match_date,
            ht.name as home_team,
            at.name as away_team,
            m.home_score_final,
            m.away_score_final,
            CASE 
                WHEN m.home_score_final > m.away_score_final THEN 'H'
                WHEN m.home_score_final < m.away_score_final THEN 'A'
                ELSE 'D'
            END as result,
            m.home_xg,
            m.away_xg,
            m.venue,
            m.attendance,
            s.year as season,
            
            -- Home team season stats (up to this match)
            hts.points as home_points,
            hts.wins as home_wins,
            hts.draws as home_draws,
            hts.losses as home_losses,
            hts.goals_for as home_gf,
            hts.goals_against as home_ga,
            hts.goal_diff as home_gd,
            
            -- Away team season stats (up to this match)
            ats.points as away_points,
            ats.wins as away_wins,
            ats.draws as away_draws,
            ats.losses as away_losses,
            ats.goals_for as away_gf,
            ats.goals_against as away_ga,
            ats.goal_diff as away_gd
            
        FROM MATCHES m
        JOIN TEAMS ht ON m.home_team_id = ht.team_id
        JOIN TEAMS at ON m.away_team_id = at.team_id
        JOIN SEASONS s ON m.season_id = s.season_id
        LEFT JOIN TEAM_SEASONS hts ON ht.team_id = hts.team_id AND s.season_id = hts.season_id
        LEFT JOIN TEAM_SEASONS ats ON at.team_id = ats.team_id AND s.season_id = ats.season_id
        WHERE m.home_score_final IS NOT NULL
        ORDER BY m.match_date
        """
        
        matches_df = pd.read_sql(query, self.conn)
        matches_df.to_csv(f'{output_dir}/match_outcomes.csv', index=False)
        print(f"✅ Exported {len(matches_df)} matches to match_outcomes.csv")
        
        # 2. Player goal scoring data
        print("\n⚽ Exporting player goal scoring data...")
        query = """
        SELECT 
            p.player_id,
            p.full_name,
            p.position,
            p.nationality,
            p.shirt_number,
            t.name as team,
            COUNT(DISTINCT COALESCE(ml.match_id, me.match_id)) as matches_played,
            COUNT(CASE WHEN me.event_type = 'Goal' THEN 1 END) as goals_scored,
            COALESCE(AVG(ml.minutes_played), 0) as avg_minutes,
            COUNT(CASE WHEN ml.is_starter = true THEN 1 END) as starts
        FROM PLAYERS p
        JOIN TEAMS t ON p.team_id = t.team_id
        LEFT JOIN MATCH_LINEUPS ml ON p.player_id = ml.player_id
        LEFT JOIN MATCH_EVENT me ON p.player_id = me.player_id
        GROUP BY p.player_id, p.full_name, p.position, p.nationality, p.shirt_number, t.name
        ORDER BY goals_scored DESC
        """
        
        players_df = pd.read_sql(query, self.conn)
        players_df.to_csv(f'{output_dir}/player_goal_stats.csv', index=False)
        print(f"✅ Exported {len(players_df)} players to player_goal_stats.csv")
        
        # 3. Team season performance data
        print("\n🏆 Exporting team season performance...")
        query = """
        SELECT 
            t.name as team,
            s.year as season,
            ts.points,
            ts.wins,
            ts.draws,
            ts.losses,
            ts.goals_for,
            ts.goals_against,
            ts.goal_diff,
            ts.position,
            l.name as league,
            l.country
        FROM TEAM_SEASONS ts
        JOIN TEAMS t ON ts.team_id = t.team_id
        JOIN SEASONS s ON ts.season_id = s.season_id
        JOIN LEAGUES l ON s.league_id = l.league_id
        ORDER BY s.year, ts.position
        """
        
        team_seasons_df = pd.read_sql(query, self.conn)
        team_seasons_df.to_csv(f'{output_dir}/team_season_performance.csv', index=False)
        print(f"✅ Exported {len(team_seasons_df)} team seasons to team_season_performance.csv")
        
        # 4. Match lineups for lineup prediction
        print("\n📋 Exporting match lineups...")
        query = """
        SELECT 
            m.match_id,
            m.match_date,
            ht.name as home_team,
            at.name as away_team,
            p.full_name as player_name,
            p.position,
            t.name as player_team,
            ml.is_starter,
            ml.minutes_played
        FROM MATCH_LINEUPS ml
        JOIN MATCHES m ON ml.match_id = m.match_id
        JOIN PLAYERS p ON ml.player_id = p.player_id
        JOIN TEAMS t ON ml.team_id = t.team_id
        JOIN TEAMS ht ON m.home_team_id = ht.team_id
        JOIN TEAMS at ON m.away_team_id = at.team_id
        ORDER BY m.match_date, ml.is_starter DESC
        """
        
        lineups_df = pd.read_sql(query, self.conn)
        lineups_df.to_csv(f'{output_dir}/match_lineups.csv', index=False)
        print(f"✅ Exported {len(lineups_df)} lineup records to match_lineups.csv")
        
        # 5. Match events for goal prediction
        print("\n🎯 Exporting match events...")
        query = """
        SELECT 
            m.match_id,
            m.match_date,
            ht.name as home_team,
            at.name as away_team,
            me.minute,
            me.event_type,
            me.xG,
            p.full_name as player_name,
            p.position,
            t.name as player_team
        FROM MATCH_EVENT me
        JOIN MATCHES m ON me.match_id = m.match_id
        JOIN PLAYERS p ON me.player_id = p.player_id
        JOIN TEAMS t ON me.team_id = t.team_id
        JOIN TEAMS ht ON m.home_team_id = ht.team_id
        JOIN TEAMS at ON m.away_team_id = at.team_id
        ORDER BY m.match_date, me.minute
        """
        
        events_df = pd.read_sql(query, self.conn)
        events_df.to_csv(f'{output_dir}/match_events.csv', index=False)
        print(f"✅ Exported {len(events_df)} events to match_events.csv")
        
        cur.close()
        
        print("\n" + "="*60)
        print("✅ ML TRAINING DATA EXPORT COMPLETE")
        print(f"📁 Files saved to: {output_dir}/")
        print("="*60)
        
        return {
            'matches': len(matches_df),
            'players': len(players_df),
            'team_seasons': len(team_seasons_df),
            'lineups': len(lineups_df),
            'events': len(events_df)
        }


def main():
    scraper = EnhancedFBrefScraper()
    
    try:
        scraper.connect_db()
        
        # Export training data from existing database
        stats = scraper.export_ml_training_data()
        
        print(f"\n📊 Training Data Summary:")
        print(f"   - Matches: {stats['matches']}")
        print(f"   - Players: {stats['players']}")
        print(f"   - Team Seasons: {stats['team_seasons']}")
        print(f"   - Lineup Records: {stats['lineups']}")
        print(f"   - Match Events: {stats['events']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        scraper.close_db()


if __name__ == '__main__':
    main()