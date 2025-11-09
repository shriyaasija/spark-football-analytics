import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { viewAPI, followAPI } from '../services/api';
import { useAuthStore } from '../store/authStore';
import type { Team, Player, Match } from '../types';
import { Heart, Users, Calendar, MapPin, Building2 } from 'lucide-react';

export default function TeamDetail() {
  const { id } = useParams();
  const { isAuthenticated } = useAuthStore();
  const [team, setTeam] = useState<Team | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [matches, setMatches] = useState<Match[]>([]);
  const [isFollowing, setIsFollowing] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (id) {
      Promise.all([
        viewAPI.getTeam(Number(id)).then(setTeam),
        viewAPI.getTeamPlayers(Number(id)).then(setPlayers),
        viewAPI.getTeamMatches(Number(id), 10).then(setMatches),
      ]).then(() => {
        setLoading(false);
      });

      if (isAuthenticated) {
        followAPI.checkFollowingTeam(Number(id)).then(setIsFollowing);
      }
    }
  }, [id, isAuthenticated]);

  const handleFollow = async () => {
    if (!id) return;
    try {
      if (isFollowing) {
        await followAPI.unfollowTeam(Number(id));
      } else {
        await followAPI.followTeam(Number(id));
      }
      setIsFollowing(!isFollowing);
    } catch (error) {
      console.error('Follow error:', error);
    }
  };

  if (loading || !team) return <div>Loading...</div>;

  const getTeamLogo = (teamName: string) => {
    const encodedName = encodeURIComponent(teamName);
    return `https://ui-avatars.com/api/?name=${encodedName}&background=random&size=200&bold=true`;
  };

  return (
    <div className="space-y-6">
      <div className="relative bg-gradient-to-br from-primary-500 via-blue-500 to-purple-500 rounded-2xl shadow-2xl overflow-hidden">
        <div className="absolute inset-0 bg-black/20"></div>
        <div className="relative p-8 text-white">
          <div className="flex flex-col md:flex-row items-center md:items-start justify-between gap-6">
            <div className="flex items-center gap-6">
              <div className="relative">
                <img
                  src={getTeamLogo(team.name)}
                  alt={team.name}
                  className="w-32 h-32 rounded-full object-cover ring-4 ring-white/50 shadow-xl"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(team.name)}&background=random&size=200`;
                  }}
                />
              </div>
              <div>
                <h1 className="text-4xl font-bold mb-2">{team.name}</h1>
                {team.city && (
                  <p className="text-lg text-white/90 flex items-center gap-2">
                    <MapPin className="h-5 w-5" />
                    {team.city}
                  </p>
                )}
                {team.stadium_name && (
                  <p className="text-md text-white/80 flex items-center gap-2 mt-1">
                    <Building2 className="h-4 w-4" />
                    {team.stadium_name}
                  </p>
                )}
              </div>
            </div>
            {isAuthenticated && (
              <button
                onClick={handleFollow}
                className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-semibold transition-all duration-300 transform hover:scale-105 ${
                  isFollowing
                    ? 'bg-white/20 text-white hover:bg-white/30 backdrop-blur-sm'
                    : 'bg-white text-primary-600 hover:bg-white/90 shadow-lg'
                }`}
              >
                <Heart className={`h-5 w-5 ${isFollowing ? 'fill-current' : ''}`} />
                <span>{isFollowing ? 'Following' : 'Follow Team'}</span>
              </button>
            )}
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold mb-4 flex items-center space-x-2">
          <Users className="h-6 w-6" />
          <span>Players</span>
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {players.map((player) => (
            <Link
              key={player.player_id}
              to={`/players/${player.player_id}`}
              className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 hover:shadow-lg transition"
            >
              <h3 className="font-semibold">{player.full_name}</h3>
              {player.position && <p className="text-sm text-gray-600 dark:text-gray-400">{player.position}</p>}
            </Link>
          ))}
        </div>
      </div>

      <div>
        <h2 className="text-2xl font-bold mb-4 flex items-center space-x-2">
          <Calendar className="h-6 w-6" />
          <span>Recent Matches</span>
        </h2>
        <div className="space-y-2">
          {matches.map((match) => (
            <Link
              key={match.match_id}
              to={`/matches/${match.match_id}`}
              className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 hover:shadow-lg transition block"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-semibold">
                    {match.home_team?.name || 'Home'} vs {match.away_team?.name || 'Away'}
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {new Date(match.match_date).toLocaleDateString()}
                  </p>
                </div>
                {match.home_score_final !== null && match.away_score_final !== null && (
                  <div className="text-2xl font-bold">
                    {match.home_score_final} - {match.away_score_final}
                  </div>
                )}
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

