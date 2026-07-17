import Foundation

/// Persists sandbox access to the user's last-opened scan folder across launches.
public enum FolderAccess {
    private static let bookmarkKey = "lastFolderBookmark"

    public static func persist(_ url: URL) {
        guard let data = try? url.bookmarkData(options: .withSecurityScope,
                                               includingResourceValuesForKeys: nil,
                                               relativeTo: nil) else { return }
        UserDefaults.standard.set(data, forKey: bookmarkKey)
    }

    /// Resolves the stored bookmark and starts security-scoped access.
    /// Access stays open for the app's lifetime by design.
    public static func restore() -> URL? {
        guard let data = UserDefaults.standard.data(forKey: bookmarkKey) else { return nil }
        var stale = false
        guard let url = try? URL(resolvingBookmarkData: data,
                                 options: .withSecurityScope,
                                 relativeTo: nil,
                                 bookmarkDataIsStale: &stale),
              url.startAccessingSecurityScopedResource()
        else { return nil }
        if stale { persist(url) }
        return url
    }
}
