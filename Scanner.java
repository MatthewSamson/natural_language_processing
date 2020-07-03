import java.io.InputStreamReader;

public class Scanner {
	private Lexer scanner = null;

	public Scanner( Lexer lexer ) {
		scanner = lexer; 
	}

	public Token getNextToken() throws java.io.IOException {
		return scanner.yylex();
	}

	public static void main(String argv[]) {
		try {
			Scanner scanner = new Scanner(new Lexer(new InputStreamReader(System.in)));
			Token tok = null;
			while( (tok=scanner.getNextToken()) != null ){
				System.out.print(tok + " ");
				if ((tok.m_type == 6) && (!tok.m_value.equals("<DOC>")) || (tok.m_type == 9))
				System.out.println();
			}
		}
		catch (Exception e) {
			System.out.println("Unexpected exception:");
			e.printStackTrace();
		}
	}
}
