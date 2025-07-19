import Navigation from "@/components/Navigation";
import Hero from "@/components/Hero";
import About from "@/components/About";
import Projects from "@/components/Projects";
import Contact from "@/components/Contact";
import Footer from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <div id="home">
        <Navigation />
        <Hero />
      </div>
      <About />
      <Projects />
      <Contact />
      <Footer />
    </div>
  );
};

export default Index;
