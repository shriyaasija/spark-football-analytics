import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { viewAPI } from '../services/api';
import type { Team } from '../types';
import { MapPin, Building2 } from 'lucide-react';

// Helper function to get team logo URL
const getTeamLogo = (teamName: string) => {
  // Using a placeholder service - you can replace with actual team logos
  const encodedName = encodeURIComponent(teamName);
  return `https://ui-avatars.com/api/?name=${encodedName}&background=random&size=128&bold=true`;
};

export default function Teams() {
  const [teams, setTeams] = useState<Team[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    viewAPI.getTeams().then((data) => {
      setTeams(data);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-primary-600 to-blue-600 bg-clip-text text-transparent mb-2">
          Premier League Teams
        </h1>
        <p className="text-gray-600 dark:text-gray-400">Explore all teams and their details</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {teams.map((team, index) => (
          <Link
            key={team.team_id}
            to={`/teams/${team.team_id}`}
            className="group relative bg-gradient-to-br from-white to-gray-50 dark:from-gray-800 dark:to-gray-900 rounded-xl shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 overflow-hidden"
            style={{ animationDelay: `${index * 50}ms` }}
          >
            {/* Gradient overlay on hover */}
            <div className="absolute inset-0 bg-gradient-to-br from-primary-500/0 to-blue-500/0 group-hover:from-primary-500/10 group-hover:to-blue-500/10 transition-all duration-300"></div>
            
            <div className="p-6 relative z-10">
              {/* Team Logo */}
              <div className="flex justify-center mb-4">
                <div className="relative">
                  <img
                    src={getTeamLogo(team.name)}
                    alt={team.name}
                    className="w-24 h-24 rounded-full object-cover ring-4 ring-primary-200 dark:ring-primary-800 group-hover:ring-primary-400 dark:group-hover:ring-primary-600 transition-all duration-300 transform group-hover:scale-110"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      target.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(team.name)}&background=random&size=128`;
                    }}
                  />
                  <div className="absolute inset-0 rounded-full bg-gradient-to-br from-primary-400/20 to-blue-400/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                </div>
              </div>
              
              {/* Team Info */}
              <div className="text-center">
                <h2 className="text-xl font-bold mb-2 text-gray-900 dark:text-white group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                  {team.name}
                </h2>
                
                {team.city && (
                  <div className="flex items-center justify-center gap-1 text-sm text-gray-600 dark:text-gray-400 mb-1">
                    <MapPin className="h-4 w-4" />
                    <span>{team.city}</span>
                  </div>
                )}
                
                {team.stadium_name && (
                  <div className="flex items-center justify-center gap-1 text-xs text-gray-500 dark:text-gray-500">
                    <Building2 className="h-3 w-3" />
                    <span>{team.stadium_name}</span>
                  </div>
                )}
              </div>
              
              {/* Hover effect indicator */}
              <div className="mt-4 flex justify-center">
                <div className="w-0 h-0.5 bg-gradient-to-r from-primary-500 to-blue-500 group-hover:w-full transition-all duration-300"></div>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}

