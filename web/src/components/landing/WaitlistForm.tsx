import { useState } from "react";
import { motion } from "framer-motion";
import { useWaitlist } from "@/hooks/useWaitlist";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";

export function WaitlistForm() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const mutation = useWaitlist();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !email.trim()) return;
    mutation.mutate({ name: name.trim(), email: email.trim() });
  };

  return (
    <section className="max-w-6xl mx-auto px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
        className="relative max-w-lg mx-auto"
      >
        {/* Glow background */}
        <div className="absolute inset-0 bg-accent-cyan/5 rounded-2xl blur-xl" />

        <div className="relative bg-bg-card border border-border-subtle rounded-2xl p-8">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-text-primary mb-2">
              Get Early Access
            </h2>
            <p className="text-text-secondary text-sm">
              Join the waitlist to be first when we open predictions to everyone.
            </p>
          </div>

          {mutation.isSuccess ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-center py-4"
            >
              <div className="text-3xl mb-3">{"\u2705"}</div>
              <p className="text-text-primary font-semibold">You're in!</p>
              <p className="text-text-secondary text-sm mt-1">
                {mutation.data?.message || "We'll be in touch soon."}
              </p>
            </motion.div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <Input
                type="text"
                placeholder="Your name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
              />
              <Input
                type="email"
                placeholder="your@email.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
              <Button
                type="submit"
                className="w-full"
                disabled={mutation.isPending}
              >
                {mutation.isPending ? "Joining..." : "Join the Waitlist"}
              </Button>
              {mutation.isError && (
                <p className="text-down text-xs text-center">
                  Something went wrong. Please try again.
                </p>
              )}
            </form>
          )}
        </div>
      </motion.div>
    </section>
  );
}
