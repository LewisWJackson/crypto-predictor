import { useState, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { PriceChart } from "@/components/chart/PriceChart";
import { useWaitlist } from "@/hooks/useWaitlist";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";

export function Hero() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const mutation = useWaitlist();
  const glowRef = useRef<HTMLDivElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim()) return;
    mutation.mutate({ name: name.trim(), email: email.trim() });
  };

  // Cursor-following glow on the chart wrapper
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!glowRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    glowRef.current.style.background = `radial-gradient(600px circle at ${x}px ${y}px, rgba(0, 212, 255, 0.07), transparent 40%)`;
  }, []);

  const handleMouseLeave = useCallback(() => {
    if (!glowRef.current) return;
    glowRef.current.style.background = "transparent";
  }, []);

  return (
    <section className="relative pt-12 pb-12 px-4">
      {/* Background glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[600px] bg-accent-cyan/5 rounded-full blur-[120px] pointer-events-none" />

      <div className="relative max-w-6xl mx-auto">
        {/* Brand */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4 }}
          className="flex items-center justify-center gap-2 mb-8"
        >
          <span className="text-accent-cyan font-bold text-sm tracking-widest uppercase">
            LJV Trading
          </span>
          <span className="text-text-secondary text-xs px-2 py-0.5 rounded-full border border-accent-cyan/20 bg-accent-cyan/5">
            Experimental
          </span>
        </motion.div>

        {/* Heading + subheading */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight mb-5">
            <span className="text-text-primary">Watch My Algorithm</span>
            <br />
            <span className="bg-gradient-to-r from-accent-cyan to-accent-blue bg-clip-text text-transparent">
              Think in Real Time
            </span>
          </h1>

          <p className="text-text-secondary text-base md:text-lg max-w-2xl mx-auto leading-relaxed">
            I'm building a machine learning model from scratch as part of my 50k to 500k challenge.
            This is the raw, unfiltered view of how it reads the market and predicts what happens next.
            It's not ready yet — but you can watch it learn.
          </p>
        </motion.div>

        {/* Inline waitlist */}
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="max-w-lg mx-auto mb-10 relative"
        >
          {/* Bright glow behind the form */}
          <div className="absolute -inset-3 rounded-2xl bg-accent-cyan/15 blur-xl pointer-events-none" />
          <div className="absolute -inset-6 rounded-3xl bg-accent-cyan/8 blur-2xl pointer-events-none" />

          <div className="relative bg-bg-card/90 backdrop-blur-sm border border-accent-cyan/30 rounded-xl p-5 shadow-[0_0_30px_rgba(0,212,255,0.12),0_0_60px_rgba(0,212,255,0.06)]">
            {mutation.isSuccess ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-center py-3"
              >
                <p className="text-accent-cyan font-semibold text-sm">You're on the list.</p>
                <p className="text-text-secondary text-xs mt-1">
                  I'll let you know when it's ready.
                </p>
              </motion.div>
            ) : (
              <>
                <p className="text-center text-text-secondary text-xs mb-3">
                  Be first to get access when it's live
                </p>
                <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3">
                  <Input
                    type="text"
                    placeholder="First name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                    className="flex-1"
                  />
                  <Input
                    type="email"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="flex-1"
                  />
                  <Button
                    type="submit"
                    disabled={mutation.isPending}
                    className="sm:w-auto whitespace-nowrap"
                  >
                    {mutation.isPending ? "Joining..." : "Get Access"}
                  </Button>
                </form>
              </>
            )}
            {mutation.isError && (
              <p className="text-down text-xs text-center mt-2">
                Something went wrong. Please try again.
              </p>
            )}
          </div>
        </motion.div>

        {/* Chart with glow */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.25 }}
          className="relative"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          {/* Static chart glow */}
          <div className="absolute -inset-1 rounded-2xl bg-gradient-to-b from-accent-cyan/10 via-transparent to-accent-cyan/5 blur-sm pointer-events-none" />
          <div className="absolute -inset-px rounded-xl bg-gradient-to-b from-accent-cyan/20 to-transparent pointer-events-none" />

          {/* Cursor-following glow overlay */}
          <div
            ref={glowRef}
            className="absolute inset-0 rounded-xl pointer-events-none z-10 transition-opacity duration-300"
          />

          <div className="relative">
            <PriceChart />
          </div>
        </motion.div>
      </div>
    </section>
  );
}
