export function Footer() {
  return (
    <footer className="max-w-6xl mx-auto px-4 py-12 border-t border-border-subtle">
      <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-text-secondary text-xs">
        <div className="font-semibold text-accent-cyan text-sm tracking-wider uppercase">
          LJV Trading
        </div>
        <p>
          This is an experimental project. Predictions are not financial advice.
          Past performance does not guarantee future results.
        </p>
      </div>
    </footer>
  );
}
