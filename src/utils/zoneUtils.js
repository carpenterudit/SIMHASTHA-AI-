export const ZONE_ORDER = ['zone_a', 'zone_b', 'zone_c', 'zone_d']

/**
 * Normalizes zone telemetry from either dictionary or array representations
 * into a predictable, canonical array of spatial quadrants:
 * [zone_a, zone_b, zone_c, zone_d].
 *
 * @param {Array|Object|null|undefined} raw - Raw zones payload from status/results API
 * @returns {Array<Object>} Normalized zone objects
 */
export function normalizeZones(raw) {
  if (!raw) return []

  let list = []
  if (Array.isArray(raw)) {
    list = raw
  } else if (typeof raw === 'object') {
    // If keyed by zone_id (e.g. { zone_a: {...}, zone_b: {...} })
    const ordered = []
    const visited = new Set()
    for (const id of ZONE_ORDER) {
      if (raw[id] && typeof raw[id] === 'object') {
        ordered.push({ ...raw[id], zone_id: raw[id].zone_id || id })
        visited.add(id)
      }
    }
    for (const [key, val] of Object.entries(raw)) {
      if (!visited.has(key) && val && typeof val === 'object') {
        ordered.push({ ...val, zone_id: val.zone_id || key })
      }
    }
    list = ordered
  }

  if (list.length === 0) return []

  // Index zones by zone_id to enforce exact canonical order: zone_a, zone_b, zone_c, zone_d
  const zoneMap = new Map()
  const extraZones = []
  for (const item of list) {
    if (item && item.zone_id) {
      zoneMap.set(item.zone_id, {
        ...item,
        capacity: item.capacity ?? item.reference_capacity ?? 10
      })
    } else if (item) {
      extraZones.push(item)
    }
  }

  const result = []
  for (const id of ZONE_ORDER) {
    if (zoneMap.has(id)) {
      result.push(zoneMap.get(id))
      zoneMap.delete(id)
    }
  }

  // Append any extra zones
  for (const item of zoneMap.values()) {
    result.push(item)
  }
  for (const item of extraZones) {
    result.push(item)
  }

  return result
}
