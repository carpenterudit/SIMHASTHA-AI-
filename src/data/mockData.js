export const dashboardData = {
  kpis: [
    { label: 'Total crowd', value: '18,420', detail: '+8.4% compared with previous hour', tone: 'blue', icon: 'Users' },
    { label: 'Crowd density', value: 'HIGH', detail: 'Current critical zones: 2', tone: 'orange', icon: 'Gauge' },
    { label: 'Active alerts', value: '03', detail: '1 Critical · 2 High', tone: 'red', icon: 'Siren' },
    { label: 'Active cameras', value: '24 / 30', detail: '24 online · 6 offline', tone: 'green', icon: 'Camera' }
  ],
  chart: [
    { time: '10:00', crowd: 11200 }, { time: '10:10', crowd: 12640 }, { time: '10:20', crowd: 11980 },
    { time: '10:30', crowd: 13740 }, { time: '10:40', crowd: 15120 }, { time: '10:50', crowd: 16480 }, { time: '11:00', crowd: 18420 }
  ],
  zones: [
    { zone: 'Ramghat', people: '7,430', capacity: '8,000', density: '93%', flow: '→ South', risk: 'Critical' },
    { zone: 'Gate B', people: '5,890', capacity: '7,500', density: '78%', flow: '→ North', risk: 'High' },
    { zone: 'Gate C', people: '2,140', capacity: '6,000', density: '36%', flow: '→ East', risk: 'Normal' }
  ],
  incidents: [
    { severity: 'Critical', location: 'Ramghat', text: 'Crowd density extremely high', time: '2 min ago' },
    { severity: 'High', location: 'Gate B', text: 'Incoming crowd increasing', time: '5 min ago' },
    { severity: 'Moderate', location: 'Parking A', text: 'Vehicle occupancy increasing', time: '8 min ago' }
  ],
  alerts: [ 'Crowd density increasing — Gate B', 'Medical request — Zone 17', 'Parking capacity 82% — Parking A' ],
  drones: [ ['D-01', 'Available'], ['D-02', 'Available'], ['D-03', 'In transit'], ['D-04', 'Offline'], ['D-05', 'Available'] ]
}
