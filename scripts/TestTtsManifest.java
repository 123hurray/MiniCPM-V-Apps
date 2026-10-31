import java.nio.file.Path;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.*;

public final class TestTtsManifest {
    private static final String NS="http://schemas.android.com/apk/res/android";
    private static String attribute(Element node,String name) { return node.getAttributeNS(NS,name); }
    private static Element verify(Document doc) {
        Element service=null, matchedFilter=null;
        NodeList services=doc.getElementsByTagName("service");
        int engines=0;
        for(int i=0;i<services.getLength();i++) {
            Element candidate=(Element)services.item(i);
            NodeList filters=candidate.getElementsByTagName("intent-filter");
            for(int j=0;j<filters.getLength();j++) {
                Element filter=(Element)filters.item(j);
                NodeList actions=filter.getElementsByTagName("action");
                for(int k=0;k<actions.getLength();k++) {
                    if(attribute((Element)actions.item(k),"name").equals("android.intent.action.TTS_SERVICE")) {
                        engines++; service=candidate; matchedFilter=filter;
                    }
                }
            }
        }
        if(engines!=1 || service==null || !"true".equals(attribute(service,"exported"))) throw new AssertionError("Exactly one exported TTS service is required");
        if(!attribute(service,"name").endsWith(".Audio8Service")) throw new AssertionError("Unexpected TTS service");
        if(!attribute(service,"permission").isEmpty()) throw new AssertionError("Do not block third-party TTS clients with a signature permission");
        boolean defaultCategory=false, metadata=false;
        NodeList categories=matchedFilter.getElementsByTagName("category");
        Element defaultElement=null;
        for(int i=0;i<categories.getLength();i++) if(attribute((Element)categories.item(i),"name").equals("android.intent.category.DEFAULT")) { defaultCategory=true; defaultElement=(Element)categories.item(i); }
        if(!defaultCategory) throw new AssertionError("System MATCH_DEFAULT_ONLY discovery requires DEFAULT on the TTS service filter");
        NodeList data=service.getElementsByTagName("meta-data");
        for(int i=0;i<data.getLength();i++) {
            Element item=(Element)data.item(i);
            if(attribute(item,"name").equals("android.speech.tts") && attribute(item,"resource").equals("@xml/tts_engine")) metadata=true;
        }
        if(!metadata) throw new AssertionError("Missing TTS metadata");
        return defaultElement;
    }
    public static void main(String[] args) throws Exception {
        var factory=DocumentBuilderFactory.newInstance(); factory.setNamespaceAware(true);
        factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl",true);
        String path=args.length==0 ? "Audio8-Android/app/src/main/AndroidManifest.xml" : args[0];
        Document doc=factory.newDocumentBuilder().parse(Path.of(path).toFile());
        Element category=verify(doc);
        category.getParentNode().removeChild(category);
        boolean rejected=false;
        try { verify(doc); } catch(AssertionError expected) { rejected=true; }
        if(!rejected) throw new AssertionError("Regression test did not reject the v4.1 missing-category bug");
        System.out.println("TTS discovery manifest checks passed, including missing-DEFAULT regression");
    }
}
