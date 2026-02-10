import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Hero } from "@/components/landing/Hero";
import { Features } from "@/components/landing/Features";
import { Footer } from "@/components/landing/Footer";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen">
        <Hero />
        <Features />
        <Footer />
      </div>
    </QueryClientProvider>
  );
}
