-- ============================================
-- USER AUTHENTICATION & ENGAGEMENT TABLES
-- ============================================

-- USERS table
CREATE TABLE IF NOT EXISTS USERS (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(150),
    is_admin BOOLEAN DEFAULT FALSE,
    profile_picture_url VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_users_username ON USERS(username);
CREATE INDEX idx_users_email ON USERS(email);

-- USER_FOLLOWS_TEAMS (M:N relationship)
CREATE TABLE IF NOT EXISTS USER_FOLLOWS_TEAMS (
    user_id INT NOT NULL,
    team_id INT NOT NULL,
    followed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, team_id),
    FOREIGN KEY (user_id) REFERENCES USERS(user_id) ON DELETE CASCADE,
    FOREIGN KEY (team_id) REFERENCES TEAMS(team_id) ON DELETE CASCADE
);

CREATE INDEX idx_user_follows_teams_user ON USER_FOLLOWS_TEAMS(user_id);
CREATE INDEX idx_user_follows_teams_team ON USER_FOLLOWS_TEAMS(team_id);

-- USER_FOLLOWS_PLAYERS (M:N relationship)
CREATE TABLE IF NOT EXISTS USER_FOLLOWS_PLAYERS (
    user_id INT NOT NULL,
    player_id INT NOT NULL,
    followed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, player_id),
    FOREIGN KEY (user_id) REFERENCES USERS(user_id) ON DELETE CASCADE,
    FOREIGN KEY (player_id) REFERENCES PLAYERS(player_id) ON DELETE CASCADE
);

CREATE INDEX idx_user_follows_players_user ON USER_FOLLOWS_PLAYERS(user_id);
CREATE INDEX idx_user_follows_players_player ON USER_FOLLOWS_PLAYERS(player_id);

-- USER_PREDICTIONS
CREATE TABLE IF NOT EXISTS USER_PREDICTIONS (
    prediction_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    season_id INT NOT NULL,
    prediction_type VARCHAR(50) NOT NULL CHECK (prediction_type IN ('season_winner', 'top_scorer', 'top_assists', 'golden_boot')),
    predicted_team_id INT,  -- For season winner
    predicted_player_id INT,  -- For top scorer/assists
    predicted_value INT,  -- Optional: predicted goals/assists count
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_correct BOOLEAN,  -- NULL until season ends, then TRUE/FALSE
    points_earned INT DEFAULT 0,
    
    FOREIGN KEY (user_id) REFERENCES USERS(user_id) ON DELETE CASCADE,
    FOREIGN KEY (season_id) REFERENCES SEASONS(season_id) ON DELETE CASCADE,
    FOREIGN KEY (predicted_team_id) REFERENCES TEAMS(team_id) ON DELETE SET NULL,
    FOREIGN KEY (predicted_player_id) REFERENCES PLAYERS(player_id) ON DELETE SET NULL,
    
    -- User can only make one prediction per type per season
    UNIQUE (user_id, season_id, prediction_type)
);

CREATE INDEX idx_predictions_user ON USER_PREDICTIONS(user_id);
CREATE INDEX idx_predictions_season ON USER_PREDICTIONS(season_id);
CREATE INDEX idx_predictions_type ON USER_PREDICTIONS(prediction_type);

-- USER_LEADERBOARD (materialized view for performance)
CREATE TABLE IF NOT EXISTS USER_LEADERBOARD (
    user_id INT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    total_predictions INT DEFAULT 0,
    correct_predictions INT DEFAULT 0,
    total_points INT DEFAULT 0,
    accuracy_percentage DECIMAL(5,2) DEFAULT 0.00,
    rank INT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES USERS(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_leaderboard_points ON USER_LEADERBOARD(total_points DESC);
CREATE INDEX idx_leaderboard_rank ON USER_LEADERBOARD(rank);

-- USER_ACTIVITY_FEED (for "Updates" tab)
CREATE TABLE IF NOT EXISTS USER_ACTIVITY_FEED (
    activity_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    activity_type VARCHAR(50) NOT NULL CHECK (activity_type IN ('match_result', 'goal', 'team_news', 'player_news')),
    related_match_id INT,
    related_team_id INT,
    related_player_id INT,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_read BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (user_id) REFERENCES USERS(user_id) ON DELETE CASCADE,
    FOREIGN KEY (related_match_id) REFERENCES MATCHES(match_id) ON DELETE CASCADE,
    FOREIGN KEY (related_team_id) REFERENCES TEAMS(team_id) ON DELETE CASCADE,
    FOREIGN KEY (related_player_id) REFERENCES PLAYERS(player_id) ON DELETE CASCADE
);

CREATE INDEX idx_activity_user ON USER_ACTIVITY_FEED(user_id);
CREATE INDEX idx_activity_created ON USER_ACTIVITY_FEED(created_at DESC);
CREATE INDEX idx_activity_unread ON USER_ACTIVITY_FEED(user_id, is_read);

-- Success message
DO $$ 
BEGIN 
    RAISE NOTICE '✅ User engagement tables created successfully!';
END $$;