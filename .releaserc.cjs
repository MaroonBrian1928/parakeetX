// Releases are cut from conventional commits on main. The version tag goes on the commit that
// triggered the run (no changelog commit: main requires CI before anything lands on it); the
// notes live on the GitHub release.
module.exports = {
  branches: ["main"],
  repositoryUrl: process.env.GITHUB_REPOSITORY
    ? `https://github.com/${process.env.GITHUB_REPOSITORY}.git`
    : undefined,
  tagFormat: "v${version}",
  plugins: [
    "@semantic-release/commit-analyzer",
    "@semantic-release/release-notes-generator",
    "@semantic-release/github",
  ],
};
