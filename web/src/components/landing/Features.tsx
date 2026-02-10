import { motion } from "framer-motion";
import { GlowCard } from "@/components/ui/GlowCard";

const features = [
  {
    title: "Custom ML Model",
    description:
      "A Temporal Fusion Transformer trained from scratch on millions of data points. No off-the-shelf indicators — this is a deep learning model that learns its own features.",
  },
  {
    title: "Live Market Feed",
    description:
      "Connected directly to Binance for real-time BTC/USDT price data. The prediction line updates with every tick, re-anchoring to the latest price.",
  },
  {
    title: "Confidence Bands",
    description:
      "Not just a single prediction — the model outputs a full probability distribution. The shaded band shows the range of likely outcomes at each step.",
  },
  {
    title: "Built in Public",
    description:
      "Part of the 50k to 500k challenge. Everything you see here is experimental. The model is still training and improving — you're seeing it in its rawest form.",
  },
];

export function Features() {
  return (
    <section className="max-w-6xl mx-auto px-4 py-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.5 }}
        className="text-center mb-12"
      >
        <h2 className="text-3xl font-bold text-text-primary mb-3">
          How It Works
        </h2>
        <p className="text-text-secondary max-w-xl mx-auto">
          A deep learning system built from the ground up to read price action
        </p>
      </motion.div>

      <div className="grid md:grid-cols-2 gap-6">
        {features.map((feature, i) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: i * 0.1 }}
          >
            <GlowCard>
              <h3 className="text-lg font-semibold text-text-primary mb-2">
                {feature.title}
              </h3>
              <p className="text-text-secondary text-sm leading-relaxed">
                {feature.description}
              </p>
            </GlowCard>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
