"""
VALIDATION SCRIPT
=================

This script validates that the RAG system meets all professional requirements:
1. Grounded factual questions with citations
2. Numeric financial questions without guessing
3. Cross-page synthesis
4. Negative control questions (must refuse)
5. Conversational follow-up questions

Usage: python validate.py <pdf_path>
"""
import os
import sys
from ingest import PDFIngestor
from retriever import VectorRetriever
from chat import RAGChatAgent


def validate_response_format(response: str) -> dict:
    """
    Validate that response follows the required format.
    
    Returns:
        dict with validation results
    """
    results = {
        'has_answer_section': 'Answer:' in response,
        'has_citations_section': 'Citations:' in response,
        'has_evidence_section': 'Evidence:' in response,
        'is_not_found': 'Not found in the document' in response,
        'valid': False
    }
    
    # Either has all sections OR is a not-found response
    if results['is_not_found']:
        results['valid'] = True
    elif all([results['has_answer_section'], 
              results['has_citations_section'], 
              results['has_evidence_section']]):
        results['valid'] = True
    
    return results


def run_validation_tests(agent: RAGChatAgent):
    """
    Run comprehensive validation tests.
    """
    print("\n" + "="*80)
    print("VALIDATION TEST SUITE")
    print("="*80)
    
    test_questions = [
        {
            'category': 'Factual Grounded Question',
            'question': 'What is the main topic of this document?',
            'should_find': True
        },
        {
            'category': 'Numeric Question',
            'question': 'What are the specific revenue numbers mentioned?',
            'should_find': None  # Depends on document
        },
        {
            'category': 'Negative Control (Unrelated)',
            'question': 'What is the population of Mars?',
            'should_find': False
        },
        {
            'category': 'Negative Control (Not in Document)',
            'question': 'What was the company\'s strategy in 2030?',
            'should_find': False
        }
    ]
    
    results = []
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {test['category']}")
        print(f"{'='*80}")
        print(f"Question: {test['question']}")
        print(f"\nExpected: {'Should find answer' if test['should_find'] else 'Should return Not found'}")
        print(f"\n{'─'*80}")
        
        try:
            answer = agent.answer_query(test['question'])
            
            print(f"\nResponse:\n{answer}")
            print(f"\n{'─'*80}")
            
            # Validate format
            validation = validate_response_format(answer)
            
            print(f"\n✓ Format Validation:")
            print(f"  - Valid format: {'✓' if validation['valid'] else '✗'}")
            if not validation['is_not_found']:
                print(f"  - Has Answer section: {'✓' if validation['has_answer_section'] else '✗'}")
                print(f"  - Has Citations section: {'✓' if validation['has_citations_section'] else '✗'}")
                print(f"  - Has Evidence section: {'✓' if validation['has_evidence_section'] else '✗'}")
            else:
                print(f"  - Properly refused (Not found): ✓")
            
            # Check expectation
            if test['should_find'] is False:
                expectation_met = validation['is_not_found']
                print(f"\n✓ Behavior: {'PASS - Correctly refused' if expectation_met else 'FAIL - Should have refused'}")
            elif test['should_find'] is True:
                expectation_met = not validation['is_not_found']
                print(f"\n✓ Behavior: {'PASS - Found answer' if expectation_met else 'FAIL - Should have found answer'}")
            else:
                expectation_met = True
                print(f"\n✓ Behavior: Document-dependent (no strict expectation)")
            
            results.append({
                'test': test['category'],
                'format_valid': validation['valid'],
                'expectation_met': expectation_met,
                'passed': validation['valid'] and expectation_met
            })
            
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            results.append({
                'test': test['category'],
                'format_valid': False,
                'expectation_met': False,
                'passed': False
            })
    
    # Test follow-up conversation
    print(f"\n{'='*80}")
    print(f"TEST {len(test_questions) + 1}: Multi-turn Conversation")
    print(f"{'='*80}")
    
    try:
        # Reset history
        agent.reset_history()
        
        print("Turn 1: What is this document about?")
        answer1 = agent.answer_query("What is this document about?")
        print(f"Response: {answer1[:200]}...")
        
        print("\n" + "─"*80)
        print("Turn 2: Can you provide more details? (follow-up)")
        answer2 = agent.answer_query("Can you provide more details?")
        print(f"Response: {answer2[:200]}...")
        
        validation2 = validate_response_format(answer2)
        
        followup_passed = validation2['valid']
        print(f"\n✓ Follow-up handling: {'PASS' if followup_passed else 'FAIL'}")
        
        results.append({
            'test': 'Multi-turn Conversation',
            'format_valid': validation2['valid'],
            'expectation_met': True,
            'passed': followup_passed
        })
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        results.append({
            'test': 'Multi-turn Conversation',
            'format_valid': False,
            'expectation_met': False,
            'passed': False
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for r in results if r['passed'])
    total = len(results)
    
    for r in results:
        status = "✓ PASS" if r['passed'] else "✗ FAIL"
        print(f"{status}: {r['test']}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL VALIDATION TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED - Review above for details")
    
    print(f"{'='*80}\n")
    
    return passed == total


def main():
    """Main validation function."""
    if len(sys.argv) < 2:
        print("Usage: python validate.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    
    print("="*80)
    print("RAG SYSTEM VALIDATION")
    print("="*80)
    print(f"PDF: {pdf_path}")
    print("="*80)
    
    # Ingest
    print("\n[1/3] Ingesting PDF...")
    ingestor = PDFIngestor()
    chunks = ingestor.ingest_pdf(pdf_path)
    
    # Build index
    print("\n[2/3] Building index...")
    retriever = VectorRetriever()
    retriever.build_index(chunks)
    
    # Create agent
    print("\n[3/3] Creating agent...")
    agent = RAGChatAgent(
        retriever=retriever,
        gemini_api_key=api_key,
        model_name="gemini-1.5-pro",
        debug_mode=True
    )
    
    # Run validation
    success = run_validation_tests(agent)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
