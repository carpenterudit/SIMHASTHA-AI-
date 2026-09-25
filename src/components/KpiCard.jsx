import * as Icons from 'lucide-react'
export default function KpiCard({ item }) { const Icon = Icons[item.icon]; return <section className={`kpi card ${item.tone}`}><div><p>{item.label}</p><h2>{item.value}</h2><small>{item.detail}</small></div><span className="kpi-icon"><Icon size={23}/></span></section> }
