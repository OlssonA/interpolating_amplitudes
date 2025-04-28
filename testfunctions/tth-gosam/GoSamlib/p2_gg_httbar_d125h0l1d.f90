module     p2_gg_httbar_d125h0l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d125h0l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   integer, private :: iv3
   integer, private :: iv4
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd125h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(63) :: acd125
      complex(ki) :: brack
      acd125(1)=dotproduct(qshift,qshift)
      acd125(2)=dotproduct(qshift,spvae1k2)
      acd125(3)=abb125(9)
      acd125(4)=dotproduct(qshift,spvae2k2)
      acd125(5)=abb125(35)
      acd125(6)=dotproduct(qshift,spval4e1)
      acd125(7)=abb125(32)
      acd125(8)=dotproduct(qshift,spval4e2)
      acd125(9)=abb125(19)
      acd125(10)=dotproduct(qshift,spval5e1)
      acd125(11)=abb125(47)
      acd125(12)=dotproduct(qshift,spval5e2)
      acd125(13)=abb125(39)
      acd125(14)=abb125(25)
      acd125(15)=dotproduct(qshift,spvae2e1)
      acd125(16)=abb125(42)
      acd125(17)=abb125(20)
      acd125(18)=abb125(41)
      acd125(19)=abb125(11)
      acd125(20)=dotproduct(qshift,spvak2e1)
      acd125(21)=abb125(14)
      acd125(22)=dotproduct(qshift,spval3e1)
      acd125(23)=abb125(22)
      acd125(24)=abb125(10)
      acd125(25)=abb125(15)
      acd125(26)=dotproduct(qshift,spvae1e2)
      acd125(27)=abb125(48)
      acd125(28)=abb125(44)
      acd125(29)=abb125(28)
      acd125(30)=abb125(50)
      acd125(31)=abb125(37)
      acd125(32)=abb125(27)
      acd125(33)=abb125(38)
      acd125(34)=dotproduct(qshift,spvae1l3)
      acd125(35)=abb125(24)
      acd125(36)=abb125(30)
      acd125(37)=abb125(18)
      acd125(38)=abb125(46)
      acd125(39)=abb125(12)
      acd125(40)=abb125(34)
      acd125(41)=abb125(17)
      acd125(42)=abb125(33)
      acd125(43)=abb125(43)
      acd125(44)=abb125(45)
      acd125(45)=abb125(26)
      acd125(46)=abb125(13)
      acd125(47)=abb125(31)
      acd125(48)=abb125(7)
      acd125(49)=abb125(8)
      acd125(50)=abb125(23)
      acd125(51)=abb125(16)
      acd125(52)=acd125(22)*acd125(30)
      acd125(53)=acd125(20)*acd125(29)
      acd125(54)=acd125(26)*acd125(31)
      acd125(55)=acd125(26)*acd125(18)
      acd125(55)=acd125(28)+acd125(55)
      acd125(55)=acd125(10)*acd125(55)
      acd125(56)=acd125(26)*acd125(16)
      acd125(56)=acd125(27)+acd125(56)
      acd125(56)=acd125(6)*acd125(56)
      acd125(57)=-acd125(1)*acd125(5)
      acd125(52)=acd125(57)+acd125(56)+acd125(55)+acd125(54)+acd125(53)-acd125(&
      &32)+acd125(52)
      acd125(52)=acd125(4)*acd125(52)
      acd125(53)=acd125(22)*acd125(23)
      acd125(54)=acd125(20)*acd125(21)
      acd125(55)=acd125(15)*acd125(24)
      acd125(56)=acd125(15)*acd125(18)
      acd125(56)=acd125(19)+acd125(56)
      acd125(56)=acd125(12)*acd125(56)
      acd125(57)=acd125(15)*acd125(16)
      acd125(57)=acd125(17)+acd125(57)
      acd125(57)=acd125(8)*acd125(57)
      acd125(58)=-acd125(1)*acd125(3)
      acd125(53)=acd125(58)+acd125(57)+acd125(56)+acd125(55)+acd125(54)-acd125(&
      &25)+acd125(53)
      acd125(53)=acd125(2)*acd125(53)
      acd125(54)=-acd125(12)*acd125(13)
      acd125(55)=-acd125(10)*acd125(11)
      acd125(56)=-acd125(8)*acd125(9)
      acd125(57)=-acd125(6)*acd125(7)
      acd125(54)=acd125(57)+acd125(56)+acd125(55)+acd125(14)+acd125(54)
      acd125(54)=acd125(1)*acd125(54)
      acd125(55)=acd125(34)*acd125(44)
      acd125(56)=acd125(15)*acd125(43)
      acd125(55)=acd125(56)-acd125(45)+acd125(55)
      acd125(55)=acd125(12)*acd125(55)
      acd125(56)=-acd125(34)*acd125(41)
      acd125(57)=acd125(26)*acd125(40)
      acd125(56)=acd125(57)-acd125(42)+acd125(56)
      acd125(56)=acd125(10)*acd125(56)
      acd125(57)=acd125(34)*acd125(38)
      acd125(58)=acd125(15)*acd125(37)
      acd125(57)=acd125(58)-acd125(39)+acd125(57)
      acd125(57)=acd125(8)*acd125(57)
      acd125(58)=-acd125(34)*acd125(35)
      acd125(59)=acd125(26)*acd125(33)
      acd125(58)=acd125(59)-acd125(36)+acd125(58)
      acd125(58)=acd125(6)*acd125(58)
      acd125(59)=-acd125(22)*acd125(47)
      acd125(60)=-acd125(20)*acd125(46)
      acd125(61)=-acd125(34)*acd125(50)
      acd125(62)=-acd125(26)*acd125(49)
      acd125(63)=-acd125(15)*acd125(48)
      brack=acd125(51)+acd125(52)+acd125(53)+acd125(54)+acd125(55)+acd125(56)+a&
      &cd125(57)+acd125(58)+acd125(59)+acd125(60)+acd125(61)+acd125(62)+acd125(&
      &63)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd125h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(83) :: acd125
      complex(ki) :: brack
      acd125(1)=qshift(iv1)
      acd125(2)=dotproduct(qshift,spvae1k2)
      acd125(3)=abb125(9)
      acd125(4)=dotproduct(qshift,spvae2k2)
      acd125(5)=abb125(35)
      acd125(6)=dotproduct(qshift,spval4e1)
      acd125(7)=abb125(32)
      acd125(8)=dotproduct(qshift,spval4e2)
      acd125(9)=abb125(19)
      acd125(10)=dotproduct(qshift,spval5e1)
      acd125(11)=abb125(47)
      acd125(12)=dotproduct(qshift,spval5e2)
      acd125(13)=abb125(39)
      acd125(14)=abb125(25)
      acd125(15)=spvae1k2(iv1)
      acd125(16)=dotproduct(qshift,qshift)
      acd125(17)=dotproduct(qshift,spvae2e1)
      acd125(18)=abb125(42)
      acd125(19)=abb125(20)
      acd125(20)=abb125(41)
      acd125(21)=abb125(11)
      acd125(22)=dotproduct(qshift,spvak2e1)
      acd125(23)=abb125(14)
      acd125(24)=dotproduct(qshift,spval3e1)
      acd125(25)=abb125(22)
      acd125(26)=abb125(10)
      acd125(27)=abb125(15)
      acd125(28)=spvae2k2(iv1)
      acd125(29)=dotproduct(qshift,spvae1e2)
      acd125(30)=abb125(48)
      acd125(31)=abb125(44)
      acd125(32)=abb125(28)
      acd125(33)=abb125(50)
      acd125(34)=abb125(37)
      acd125(35)=abb125(27)
      acd125(36)=spval4e1(iv1)
      acd125(37)=abb125(38)
      acd125(38)=dotproduct(qshift,spvae1l3)
      acd125(39)=abb125(24)
      acd125(40)=abb125(30)
      acd125(41)=spval4e2(iv1)
      acd125(42)=abb125(18)
      acd125(43)=abb125(46)
      acd125(44)=abb125(12)
      acd125(45)=spval5e1(iv1)
      acd125(46)=abb125(34)
      acd125(47)=abb125(17)
      acd125(48)=abb125(33)
      acd125(49)=spval5e2(iv1)
      acd125(50)=abb125(43)
      acd125(51)=abb125(45)
      acd125(52)=abb125(26)
      acd125(53)=spvak2e1(iv1)
      acd125(54)=abb125(13)
      acd125(55)=spval3e1(iv1)
      acd125(56)=abb125(31)
      acd125(57)=spvae2e1(iv1)
      acd125(58)=abb125(7)
      acd125(59)=spvae1e2(iv1)
      acd125(60)=abb125(8)
      acd125(61)=spvae1l3(iv1)
      acd125(62)=abb125(23)
      acd125(63)=acd125(10)*acd125(20)
      acd125(64)=acd125(6)*acd125(18)
      acd125(63)=acd125(34)+acd125(63)+acd125(64)
      acd125(64)=acd125(59)*acd125(63)
      acd125(65)=acd125(20)*acd125(45)
      acd125(66)=acd125(18)*acd125(36)
      acd125(65)=acd125(65)+acd125(66)
      acd125(65)=acd125(29)*acd125(65)
      acd125(66)=acd125(55)*acd125(33)
      acd125(67)=acd125(53)*acd125(32)
      acd125(68)=acd125(45)*acd125(31)
      acd125(69)=acd125(36)*acd125(30)
      acd125(70)=2.0_ki*acd125(1)
      acd125(71)=-acd125(5)*acd125(70)
      acd125(64)=acd125(71)+acd125(65)+acd125(69)+acd125(68)+acd125(66)+acd125(&
      &67)+acd125(64)
      acd125(64)=acd125(4)*acd125(64)
      acd125(65)=acd125(12)*acd125(20)
      acd125(66)=acd125(8)*acd125(18)
      acd125(65)=acd125(26)+acd125(65)+acd125(66)
      acd125(66)=acd125(57)*acd125(65)
      acd125(67)=acd125(20)*acd125(49)
      acd125(68)=acd125(18)*acd125(41)
      acd125(67)=acd125(67)+acd125(68)
      acd125(67)=acd125(17)*acd125(67)
      acd125(68)=acd125(55)*acd125(25)
      acd125(69)=acd125(53)*acd125(23)
      acd125(71)=acd125(49)*acd125(21)
      acd125(72)=acd125(41)*acd125(19)
      acd125(73)=-acd125(3)*acd125(70)
      acd125(66)=acd125(73)+acd125(67)+acd125(72)+acd125(71)+acd125(68)+acd125(&
      &69)+acd125(66)
      acd125(66)=acd125(2)*acd125(66)
      acd125(63)=acd125(29)*acd125(63)
      acd125(67)=acd125(24)*acd125(33)
      acd125(68)=acd125(22)*acd125(32)
      acd125(69)=-acd125(16)*acd125(5)
      acd125(71)=acd125(10)*acd125(31)
      acd125(72)=acd125(6)*acd125(30)
      acd125(63)=acd125(63)+acd125(72)+acd125(71)+acd125(69)+acd125(68)-acd125(&
      &35)+acd125(67)
      acd125(63)=acd125(28)*acd125(63)
      acd125(65)=acd125(17)*acd125(65)
      acd125(67)=acd125(24)*acd125(25)
      acd125(68)=acd125(22)*acd125(23)
      acd125(69)=-acd125(16)*acd125(3)
      acd125(71)=acd125(12)*acd125(21)
      acd125(72)=acd125(8)*acd125(19)
      acd125(65)=acd125(65)+acd125(72)+acd125(71)+acd125(69)+acd125(68)-acd125(&
      &27)+acd125(67)
      acd125(65)=acd125(15)*acd125(65)
      acd125(67)=-acd125(49)*acd125(13)
      acd125(68)=-acd125(45)*acd125(11)
      acd125(69)=-acd125(41)*acd125(9)
      acd125(71)=-acd125(36)*acd125(7)
      acd125(67)=acd125(71)+acd125(69)+acd125(67)+acd125(68)
      acd125(67)=acd125(16)*acd125(67)
      acd125(68)=-acd125(12)*acd125(13)
      acd125(69)=-acd125(10)*acd125(11)
      acd125(71)=-acd125(8)*acd125(9)
      acd125(72)=-acd125(6)*acd125(7)
      acd125(68)=acd125(72)+acd125(71)+acd125(69)+acd125(14)+acd125(68)
      acd125(68)=acd125(68)*acd125(70)
      acd125(69)=acd125(61)*acd125(51)
      acd125(70)=acd125(57)*acd125(50)
      acd125(69)=acd125(69)+acd125(70)
      acd125(69)=acd125(12)*acd125(69)
      acd125(70)=-acd125(61)*acd125(47)
      acd125(71)=acd125(59)*acd125(46)
      acd125(70)=acd125(70)+acd125(71)
      acd125(70)=acd125(10)*acd125(70)
      acd125(71)=acd125(61)*acd125(43)
      acd125(72)=acd125(57)*acd125(42)
      acd125(71)=acd125(71)+acd125(72)
      acd125(71)=acd125(8)*acd125(71)
      acd125(72)=-acd125(61)*acd125(39)
      acd125(73)=acd125(59)*acd125(37)
      acd125(72)=acd125(72)+acd125(73)
      acd125(72)=acd125(6)*acd125(72)
      acd125(73)=acd125(45)*acd125(46)
      acd125(74)=acd125(36)*acd125(37)
      acd125(73)=acd125(73)+acd125(74)
      acd125(73)=acd125(29)*acd125(73)
      acd125(74)=acd125(49)*acd125(50)
      acd125(75)=acd125(41)*acd125(42)
      acd125(74)=acd125(74)+acd125(75)
      acd125(74)=acd125(17)*acd125(74)
      acd125(75)=-acd125(55)*acd125(56)
      acd125(76)=-acd125(53)*acd125(54)
      acd125(77)=-acd125(61)*acd125(62)
      acd125(78)=-acd125(59)*acd125(60)
      acd125(79)=-acd125(57)*acd125(58)
      acd125(80)=acd125(38)*acd125(51)
      acd125(80)=-acd125(52)+acd125(80)
      acd125(80)=acd125(49)*acd125(80)
      acd125(81)=-acd125(38)*acd125(47)
      acd125(81)=-acd125(48)+acd125(81)
      acd125(81)=acd125(45)*acd125(81)
      acd125(82)=acd125(38)*acd125(43)
      acd125(82)=-acd125(44)+acd125(82)
      acd125(82)=acd125(41)*acd125(82)
      acd125(83)=-acd125(38)*acd125(39)
      acd125(83)=-acd125(40)+acd125(83)
      acd125(83)=acd125(36)*acd125(83)
      brack=acd125(63)+acd125(64)+acd125(65)+acd125(66)+acd125(67)+acd125(68)+a&
      &cd125(69)+acd125(70)+acd125(71)+acd125(72)+acd125(73)+acd125(74)+acd125(&
      &75)+acd125(76)+acd125(77)+acd125(78)+acd125(79)+acd125(80)+acd125(81)+ac&
      &d125(82)+acd125(83)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd125h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(84) :: acd125
      complex(ki) :: brack
      acd125(1)=d(iv1,iv2)
      acd125(2)=dotproduct(qshift,spvae1k2)
      acd125(3)=abb125(9)
      acd125(4)=dotproduct(qshift,spvae2k2)
      acd125(5)=abb125(35)
      acd125(6)=dotproduct(qshift,spval4e1)
      acd125(7)=abb125(32)
      acd125(8)=dotproduct(qshift,spval4e2)
      acd125(9)=abb125(19)
      acd125(10)=dotproduct(qshift,spval5e1)
      acd125(11)=abb125(47)
      acd125(12)=dotproduct(qshift,spval5e2)
      acd125(13)=abb125(39)
      acd125(14)=abb125(25)
      acd125(15)=qshift(iv1)
      acd125(16)=spvae1k2(iv2)
      acd125(17)=spvae2k2(iv2)
      acd125(18)=spval4e1(iv2)
      acd125(19)=spval4e2(iv2)
      acd125(20)=spval5e1(iv2)
      acd125(21)=spval5e2(iv2)
      acd125(22)=qshift(iv2)
      acd125(23)=spvae1k2(iv1)
      acd125(24)=spvae2k2(iv1)
      acd125(25)=spval4e1(iv1)
      acd125(26)=spval4e2(iv1)
      acd125(27)=spval5e1(iv1)
      acd125(28)=spval5e2(iv1)
      acd125(29)=dotproduct(qshift,spvae2e1)
      acd125(30)=abb125(42)
      acd125(31)=abb125(20)
      acd125(32)=abb125(41)
      acd125(33)=abb125(11)
      acd125(34)=spvak2e1(iv2)
      acd125(35)=abb125(14)
      acd125(36)=spval3e1(iv2)
      acd125(37)=abb125(22)
      acd125(38)=spvae2e1(iv2)
      acd125(39)=abb125(10)
      acd125(40)=spvak2e1(iv1)
      acd125(41)=spval3e1(iv1)
      acd125(42)=spvae2e1(iv1)
      acd125(43)=dotproduct(qshift,spvae1e2)
      acd125(44)=abb125(48)
      acd125(45)=abb125(44)
      acd125(46)=abb125(28)
      acd125(47)=abb125(50)
      acd125(48)=spvae1e2(iv2)
      acd125(49)=abb125(37)
      acd125(50)=spvae1e2(iv1)
      acd125(51)=abb125(38)
      acd125(52)=spvae1l3(iv2)
      acd125(53)=abb125(24)
      acd125(54)=spvae1l3(iv1)
      acd125(55)=abb125(18)
      acd125(56)=abb125(46)
      acd125(57)=abb125(34)
      acd125(58)=abb125(17)
      acd125(59)=abb125(43)
      acd125(60)=abb125(45)
      acd125(61)=-acd125(15)*acd125(16)
      acd125(62)=-acd125(22)*acd125(23)
      acd125(63)=-acd125(2)*acd125(1)
      acd125(61)=acd125(63)+acd125(61)+acd125(62)
      acd125(61)=acd125(3)*acd125(61)
      acd125(62)=-acd125(15)*acd125(17)
      acd125(63)=-acd125(22)*acd125(24)
      acd125(64)=-acd125(4)*acd125(1)
      acd125(62)=acd125(64)+acd125(62)+acd125(63)
      acd125(62)=acd125(5)*acd125(62)
      acd125(63)=-acd125(18)*acd125(15)
      acd125(64)=-acd125(25)*acd125(22)
      acd125(65)=-acd125(6)*acd125(1)
      acd125(63)=acd125(65)+acd125(63)+acd125(64)
      acd125(63)=acd125(7)*acd125(63)
      acd125(64)=-acd125(19)*acd125(15)
      acd125(65)=-acd125(26)*acd125(22)
      acd125(66)=-acd125(8)*acd125(1)
      acd125(64)=acd125(66)+acd125(64)+acd125(65)
      acd125(64)=acd125(9)*acd125(64)
      acd125(65)=-acd125(20)*acd125(15)
      acd125(66)=-acd125(27)*acd125(22)
      acd125(67)=-acd125(10)*acd125(1)
      acd125(65)=acd125(67)+acd125(65)+acd125(66)
      acd125(65)=acd125(11)*acd125(65)
      acd125(66)=-acd125(21)*acd125(15)
      acd125(67)=-acd125(28)*acd125(22)
      acd125(68)=-acd125(12)*acd125(1)
      acd125(66)=acd125(68)+acd125(66)+acd125(67)
      acd125(66)=acd125(13)*acd125(66)
      acd125(67)=acd125(14)*acd125(1)
      acd125(61)=acd125(63)+acd125(64)+acd125(65)+acd125(66)+acd125(67)+acd125(&
      &61)+acd125(62)
      acd125(62)=acd125(8)*acd125(30)
      acd125(63)=acd125(12)*acd125(32)
      acd125(62)=acd125(39)+acd125(63)+acd125(62)
      acd125(63)=acd125(38)*acd125(23)
      acd125(64)=acd125(42)*acd125(16)
      acd125(63)=acd125(63)+acd125(64)
      acd125(62)=acd125(63)*acd125(62)
      acd125(63)=acd125(6)*acd125(30)
      acd125(64)=acd125(10)*acd125(32)
      acd125(63)=acd125(49)+acd125(64)+acd125(63)
      acd125(64)=acd125(48)*acd125(24)
      acd125(65)=acd125(50)*acd125(17)
      acd125(64)=acd125(64)+acd125(65)
      acd125(63)=acd125(64)*acd125(63)
      acd125(64)=acd125(19)*acd125(30)
      acd125(65)=acd125(21)*acd125(32)
      acd125(64)=acd125(64)+acd125(65)
      acd125(65)=acd125(42)*acd125(64)
      acd125(66)=acd125(26)*acd125(38)
      acd125(67)=acd125(30)*acd125(66)
      acd125(68)=acd125(28)*acd125(38)
      acd125(69)=acd125(32)*acd125(68)
      acd125(65)=acd125(69)+acd125(67)+acd125(65)
      acd125(65)=acd125(2)*acd125(65)
      acd125(67)=acd125(18)*acd125(30)
      acd125(69)=acd125(20)*acd125(32)
      acd125(67)=acd125(67)+acd125(69)
      acd125(69)=acd125(50)*acd125(67)
      acd125(70)=acd125(25)*acd125(48)
      acd125(71)=acd125(30)*acd125(70)
      acd125(72)=acd125(27)*acd125(48)
      acd125(73)=acd125(32)*acd125(72)
      acd125(69)=acd125(73)+acd125(71)+acd125(69)
      acd125(69)=acd125(4)*acd125(69)
      acd125(64)=acd125(23)*acd125(64)
      acd125(71)=acd125(26)*acd125(16)
      acd125(73)=acd125(30)*acd125(71)
      acd125(74)=acd125(28)*acd125(16)
      acd125(75)=acd125(32)*acd125(74)
      acd125(64)=acd125(75)+acd125(73)+acd125(64)
      acd125(64)=acd125(29)*acd125(64)
      acd125(67)=acd125(24)*acd125(67)
      acd125(73)=acd125(25)*acd125(17)
      acd125(75)=acd125(30)*acd125(73)
      acd125(76)=acd125(27)*acd125(17)
      acd125(77)=acd125(32)*acd125(76)
      acd125(67)=acd125(77)+acd125(75)+acd125(67)
      acd125(67)=acd125(43)*acd125(67)
      acd125(75)=acd125(34)*acd125(23)
      acd125(77)=acd125(40)*acd125(16)
      acd125(75)=acd125(77)+acd125(75)
      acd125(75)=acd125(35)*acd125(75)
      acd125(77)=acd125(36)*acd125(23)
      acd125(78)=acd125(41)*acd125(16)
      acd125(77)=acd125(78)+acd125(77)
      acd125(77)=acd125(37)*acd125(77)
      acd125(78)=acd125(19)*acd125(23)
      acd125(71)=acd125(78)+acd125(71)
      acd125(71)=acd125(31)*acd125(71)
      acd125(78)=acd125(21)*acd125(23)
      acd125(74)=acd125(78)+acd125(74)
      acd125(74)=acd125(33)*acd125(74)
      acd125(78)=acd125(18)*acd125(24)
      acd125(73)=acd125(78)+acd125(73)
      acd125(73)=acd125(44)*acd125(73)
      acd125(78)=acd125(20)*acd125(24)
      acd125(76)=acd125(78)+acd125(76)
      acd125(76)=acd125(45)*acd125(76)
      acd125(78)=acd125(34)*acd125(24)
      acd125(79)=acd125(40)*acd125(17)
      acd125(78)=acd125(78)+acd125(79)
      acd125(78)=acd125(46)*acd125(78)
      acd125(79)=acd125(36)*acd125(24)
      acd125(80)=acd125(41)*acd125(17)
      acd125(79)=acd125(79)+acd125(80)
      acd125(79)=acd125(47)*acd125(79)
      acd125(80)=acd125(18)*acd125(50)
      acd125(70)=acd125(80)+acd125(70)
      acd125(70)=acd125(51)*acd125(70)
      acd125(80)=-acd125(52)*acd125(25)
      acd125(81)=-acd125(54)*acd125(18)
      acd125(80)=acd125(80)+acd125(81)
      acd125(80)=acd125(53)*acd125(80)
      acd125(81)=acd125(19)*acd125(42)
      acd125(66)=acd125(81)+acd125(66)
      acd125(66)=acd125(55)*acd125(66)
      acd125(81)=acd125(52)*acd125(26)
      acd125(82)=acd125(54)*acd125(19)
      acd125(81)=acd125(81)+acd125(82)
      acd125(81)=acd125(56)*acd125(81)
      acd125(82)=acd125(20)*acd125(50)
      acd125(72)=acd125(82)+acd125(72)
      acd125(72)=acd125(57)*acd125(72)
      acd125(82)=-acd125(52)*acd125(27)
      acd125(83)=-acd125(54)*acd125(20)
      acd125(82)=acd125(82)+acd125(83)
      acd125(82)=acd125(58)*acd125(82)
      acd125(83)=acd125(21)*acd125(42)
      acd125(68)=acd125(83)+acd125(68)
      acd125(68)=acd125(59)*acd125(68)
      acd125(83)=acd125(52)*acd125(28)
      acd125(84)=acd125(54)*acd125(21)
      acd125(83)=acd125(83)+acd125(84)
      acd125(83)=acd125(60)*acd125(83)
      brack=2.0_ki*acd125(61)+acd125(62)+acd125(63)+acd125(64)+acd125(65)+acd12&
      &5(66)+acd125(67)+acd125(68)+acd125(69)+acd125(70)+acd125(71)+acd125(72)+&
      &acd125(73)+acd125(74)+acd125(75)+acd125(76)+acd125(77)+acd125(78)+acd125&
      &(79)+acd125(80)+acd125(81)+acd125(82)+acd125(83)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd125h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(49) :: acd125
      complex(ki) :: brack
      acd125(1)=d(iv1,iv2)
      acd125(2)=spvae1k2(iv3)
      acd125(3)=abb125(9)
      acd125(4)=spvae2k2(iv3)
      acd125(5)=abb125(35)
      acd125(6)=spval4e1(iv3)
      acd125(7)=abb125(32)
      acd125(8)=spval4e2(iv3)
      acd125(9)=abb125(19)
      acd125(10)=spval5e1(iv3)
      acd125(11)=abb125(47)
      acd125(12)=spval5e2(iv3)
      acd125(13)=abb125(39)
      acd125(14)=d(iv1,iv3)
      acd125(15)=spvae1k2(iv2)
      acd125(16)=spvae2k2(iv2)
      acd125(17)=spval4e1(iv2)
      acd125(18)=spval4e2(iv2)
      acd125(19)=spval5e1(iv2)
      acd125(20)=spval5e2(iv2)
      acd125(21)=d(iv2,iv3)
      acd125(22)=spvae1k2(iv1)
      acd125(23)=spvae2k2(iv1)
      acd125(24)=spval4e1(iv1)
      acd125(25)=spval4e2(iv1)
      acd125(26)=spval5e1(iv1)
      acd125(27)=spval5e2(iv1)
      acd125(28)=spvae2e1(iv3)
      acd125(29)=abb125(42)
      acd125(30)=spvae2e1(iv2)
      acd125(31)=abb125(41)
      acd125(32)=spvae2e1(iv1)
      acd125(33)=spvae1e2(iv3)
      acd125(34)=spvae1e2(iv2)
      acd125(35)=spvae1e2(iv1)
      acd125(36)=-acd125(7)*acd125(6)
      acd125(37)=-acd125(9)*acd125(8)
      acd125(38)=-acd125(11)*acd125(10)
      acd125(39)=-acd125(13)*acd125(12)
      acd125(36)=acd125(39)+acd125(38)+acd125(37)+acd125(36)
      acd125(37)=2.0_ki*acd125(1)
      acd125(36)=acd125(37)*acd125(36)
      acd125(37)=-acd125(2)*acd125(1)
      acd125(38)=-acd125(15)*acd125(14)
      acd125(39)=-acd125(22)*acd125(21)
      acd125(37)=acd125(39)+acd125(37)+acd125(38)
      acd125(37)=acd125(3)*acd125(37)
      acd125(38)=-acd125(4)*acd125(1)
      acd125(39)=-acd125(16)*acd125(14)
      acd125(40)=-acd125(23)*acd125(21)
      acd125(38)=acd125(40)+acd125(38)+acd125(39)
      acd125(38)=acd125(5)*acd125(38)
      acd125(37)=acd125(37)+acd125(38)
      acd125(38)=acd125(8)*acd125(29)
      acd125(39)=acd125(12)*acd125(31)
      acd125(38)=acd125(39)+acd125(38)
      acd125(39)=acd125(30)*acd125(22)
      acd125(40)=acd125(32)*acd125(15)
      acd125(39)=acd125(39)+acd125(40)
      acd125(38)=acd125(39)*acd125(38)
      acd125(39)=acd125(6)*acd125(29)
      acd125(40)=acd125(10)*acd125(31)
      acd125(39)=acd125(40)+acd125(39)
      acd125(40)=acd125(34)*acd125(23)
      acd125(41)=acd125(35)*acd125(16)
      acd125(40)=acd125(40)+acd125(41)
      acd125(39)=acd125(40)*acd125(39)
      acd125(40)=acd125(33)*acd125(23)
      acd125(41)=acd125(35)*acd125(4)
      acd125(40)=acd125(40)+acd125(41)
      acd125(41)=acd125(29)*acd125(40)
      acd125(42)=2.0_ki*acd125(14)
      acd125(43)=-acd125(7)*acd125(42)
      acd125(41)=acd125(43)+acd125(41)
      acd125(41)=acd125(17)*acd125(41)
      acd125(43)=acd125(28)*acd125(22)
      acd125(44)=acd125(32)*acd125(2)
      acd125(43)=acd125(43)+acd125(44)
      acd125(44)=acd125(29)*acd125(43)
      acd125(45)=-acd125(9)*acd125(42)
      acd125(44)=acd125(45)+acd125(44)
      acd125(44)=acd125(18)*acd125(44)
      acd125(40)=acd125(31)*acd125(40)
      acd125(45)=-acd125(11)*acd125(42)
      acd125(40)=acd125(45)+acd125(40)
      acd125(40)=acd125(19)*acd125(40)
      acd125(43)=acd125(31)*acd125(43)
      acd125(42)=-acd125(13)*acd125(42)
      acd125(42)=acd125(42)+acd125(43)
      acd125(42)=acd125(20)*acd125(42)
      acd125(43)=acd125(33)*acd125(16)
      acd125(45)=acd125(34)*acd125(4)
      acd125(43)=acd125(43)+acd125(45)
      acd125(45)=acd125(29)*acd125(43)
      acd125(46)=2.0_ki*acd125(21)
      acd125(47)=-acd125(7)*acd125(46)
      acd125(45)=acd125(47)+acd125(45)
      acd125(45)=acd125(24)*acd125(45)
      acd125(47)=acd125(28)*acd125(15)
      acd125(48)=acd125(30)*acd125(2)
      acd125(47)=acd125(47)+acd125(48)
      acd125(48)=acd125(29)*acd125(47)
      acd125(49)=-acd125(9)*acd125(46)
      acd125(48)=acd125(49)+acd125(48)
      acd125(48)=acd125(25)*acd125(48)
      acd125(43)=acd125(31)*acd125(43)
      acd125(49)=-acd125(11)*acd125(46)
      acd125(43)=acd125(49)+acd125(43)
      acd125(43)=acd125(26)*acd125(43)
      acd125(47)=acd125(31)*acd125(47)
      acd125(46)=-acd125(13)*acd125(46)
      acd125(46)=acd125(46)+acd125(47)
      acd125(46)=acd125(27)*acd125(46)
      brack=acd125(36)+2.0_ki*acd125(37)+acd125(38)+acd125(39)+acd125(40)+acd12&
      &5(41)+acd125(42)+acd125(43)+acd125(44)+acd125(45)+acd125(46)+acd125(48)
   end function brack_4
!---#] function brack_4:
!---#[ function brack_5:
   pure function brack_5(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd125h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd125
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_5
!---#] function brack_5:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3,i4) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd125h0
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      integer, intent(in), optional :: i4
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k3-k2+k5
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      if(present(i3)) then
          iv3=i3
          deg=3
      else
          iv3=1
      end if
      if(present(i4)) then
          iv4=i4
          deg=4
      else
          iv4=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
      if(deg.eq.3) then
         numerator = cond(epspow.eq.t1,brack_4,Q,mu2)
         return
      end if
      if(deg.eq.4) then
         numerator = cond(epspow.eq.t1,brack_5,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p2_gg_httbar_d125h0l1d
