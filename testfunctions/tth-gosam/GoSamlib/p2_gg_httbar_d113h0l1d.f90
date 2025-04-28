module     p2_gg_httbar_d113h0l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d113h0l1d.f90
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
      use p2_gg_httbar_abbrevd113h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(69) :: acd113
      complex(ki) :: brack
      acd113(1)=dotproduct(k2,qshift)
      acd113(2)=dotproduct(qshift,spval4e1)
      acd113(3)=abb113(48)
      acd113(4)=dotproduct(qshift,spval5e1)
      acd113(5)=abb113(44)
      acd113(6)=dotproduct(qshift,spvae2e1)
      acd113(7)=abb113(37)
      acd113(8)=abb113(25)
      acd113(9)=dotproduct(qshift,qshift)
      acd113(10)=abb113(23)
      acd113(11)=abb113(45)
      acd113(12)=abb113(33)
      acd113(13)=dotproduct(qshift,spvae1k2)
      acd113(14)=abb113(38)
      acd113(15)=dotproduct(qshift,spvae1e2)
      acd113(16)=abb113(20)
      acd113(17)=abb113(19)
      acd113(18)=dotproduct(qshift,spvae2k2)
      acd113(19)=abb113(47)
      acd113(20)=abb113(49)
      acd113(21)=dotproduct(qshift,spval3e2)
      acd113(22)=abb113(11)
      acd113(23)=abb113(7)
      acd113(24)=abb113(42)
      acd113(25)=abb113(46)
      acd113(26)=abb113(13)
      acd113(27)=abb113(8)
      acd113(28)=dotproduct(qshift,spval4e2)
      acd113(29)=dotproduct(qshift,spval5e2)
      acd113(30)=abb113(22)
      acd113(31)=abb113(15)
      acd113(32)=abb113(43)
      acd113(33)=abb113(30)
      acd113(34)=abb113(27)
      acd113(35)=dotproduct(qshift,spval4k2)
      acd113(36)=abb113(10)
      acd113(37)=dotproduct(qshift,spval5k2)
      acd113(38)=abb113(9)
      acd113(39)=abb113(34)
      acd113(40)=dotproduct(qshift,spvae2l3)
      acd113(41)=abb113(32)
      acd113(42)=abb113(14)
      acd113(43)=abb113(35)
      acd113(44)=abb113(18)
      acd113(45)=abb113(21)
      acd113(46)=abb113(50)
      acd113(47)=abb113(12)
      acd113(48)=abb113(31)
      acd113(49)=abb113(29)
      acd113(50)=abb113(28)
      acd113(51)=abb113(16)
      acd113(52)=abb113(26)
      acd113(53)=abb113(24)
      acd113(54)=abb113(39)
      acd113(55)=abb113(17)
      acd113(56)=acd113(40)*acd113(46)
      acd113(57)=acd113(37)*acd113(44)
      acd113(58)=acd113(35)*acd113(43)
      acd113(59)=acd113(18)*acd113(45)
      acd113(60)=-acd113(9)*acd113(16)
      acd113(61)=acd113(18)*acd113(24)
      acd113(61)=acd113(25)+acd113(61)
      acd113(61)=acd113(4)*acd113(61)
      acd113(62)=acd113(18)*acd113(19)
      acd113(62)=acd113(20)+acd113(62)
      acd113(62)=acd113(2)*acd113(62)
      acd113(56)=acd113(62)+acd113(61)+acd113(60)+acd113(59)+acd113(58)+acd113(&
      &57)-acd113(47)+acd113(56)
      acd113(56)=acd113(15)*acd113(56)
      acd113(57)=acd113(29)*acd113(24)
      acd113(58)=acd113(28)*acd113(19)
      acd113(57)=acd113(58)+acd113(30)+acd113(57)
      acd113(57)=acd113(13)*acd113(57)
      acd113(58)=acd113(29)*acd113(32)
      acd113(59)=acd113(28)*acd113(31)
      acd113(60)=acd113(21)*acd113(33)
      acd113(61)=acd113(1)*acd113(7)
      acd113(62)=acd113(9)*acd113(12)
      acd113(57)=acd113(57)+acd113(62)+acd113(61)+acd113(60)+acd113(59)-acd113(&
      &34)+acd113(58)
      acd113(57)=acd113(6)*acd113(57)
      acd113(58)=acd113(40)*acd113(41)
      acd113(59)=acd113(37)*acd113(38)
      acd113(60)=acd113(35)*acd113(36)
      acd113(61)=acd113(18)*acd113(39)
      acd113(62)=-acd113(9)*acd113(14)
      acd113(58)=acd113(62)+acd113(61)+acd113(60)+acd113(59)-acd113(42)+acd113(&
      &58)
      acd113(58)=acd113(13)*acd113(58)
      acd113(59)=acd113(21)*acd113(26)
      acd113(60)=acd113(1)*acd113(5)
      acd113(61)=-acd113(9)*acd113(11)
      acd113(59)=acd113(61)+acd113(60)-acd113(27)+acd113(59)
      acd113(59)=acd113(4)*acd113(59)
      acd113(60)=acd113(21)*acd113(22)
      acd113(61)=acd113(1)*acd113(3)
      acd113(62)=-acd113(9)*acd113(10)
      acd113(60)=acd113(62)+acd113(61)-acd113(23)+acd113(60)
      acd113(60)=acd113(2)*acd113(60)
      acd113(61)=-acd113(40)*acd113(51)
      acd113(62)=-acd113(37)*acd113(49)
      acd113(63)=-acd113(35)*acd113(48)
      acd113(64)=-acd113(29)*acd113(53)
      acd113(65)=-acd113(28)*acd113(52)
      acd113(66)=-acd113(21)*acd113(54)
      acd113(67)=-acd113(1)*acd113(8)
      acd113(68)=-acd113(18)*acd113(50)
      acd113(69)=acd113(9)*acd113(17)
      brack=acd113(55)+acd113(56)+acd113(57)+acd113(58)+acd113(59)+acd113(60)+a&
      &cd113(61)+acd113(62)+acd113(63)+acd113(64)+acd113(65)+acd113(66)+acd113(&
      &67)+acd113(68)+acd113(69)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd113h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(87) :: acd113
      complex(ki) :: brack
      acd113(1)=k2(iv1)
      acd113(2)=dotproduct(qshift,spval4e1)
      acd113(3)=abb113(48)
      acd113(4)=dotproduct(qshift,spval5e1)
      acd113(5)=abb113(44)
      acd113(6)=dotproduct(qshift,spvae2e1)
      acd113(7)=abb113(37)
      acd113(8)=abb113(25)
      acd113(9)=qshift(iv1)
      acd113(10)=abb113(23)
      acd113(11)=abb113(45)
      acd113(12)=abb113(33)
      acd113(13)=dotproduct(qshift,spvae1k2)
      acd113(14)=abb113(38)
      acd113(15)=dotproduct(qshift,spvae1e2)
      acd113(16)=abb113(20)
      acd113(17)=abb113(19)
      acd113(18)=spval4e1(iv1)
      acd113(19)=dotproduct(k2,qshift)
      acd113(20)=dotproduct(qshift,qshift)
      acd113(21)=dotproduct(qshift,spvae2k2)
      acd113(22)=abb113(47)
      acd113(23)=abb113(49)
      acd113(24)=dotproduct(qshift,spval3e2)
      acd113(25)=abb113(11)
      acd113(26)=abb113(7)
      acd113(27)=spval5e1(iv1)
      acd113(28)=abb113(42)
      acd113(29)=abb113(46)
      acd113(30)=abb113(13)
      acd113(31)=abb113(8)
      acd113(32)=spvae2e1(iv1)
      acd113(33)=dotproduct(qshift,spval4e2)
      acd113(34)=dotproduct(qshift,spval5e2)
      acd113(35)=abb113(22)
      acd113(36)=abb113(15)
      acd113(37)=abb113(43)
      acd113(38)=abb113(30)
      acd113(39)=abb113(27)
      acd113(40)=spvae1k2(iv1)
      acd113(41)=dotproduct(qshift,spval4k2)
      acd113(42)=abb113(10)
      acd113(43)=dotproduct(qshift,spval5k2)
      acd113(44)=abb113(9)
      acd113(45)=abb113(34)
      acd113(46)=dotproduct(qshift,spvae2l3)
      acd113(47)=abb113(32)
      acd113(48)=abb113(14)
      acd113(49)=spvae1e2(iv1)
      acd113(50)=abb113(35)
      acd113(51)=abb113(18)
      acd113(52)=abb113(21)
      acd113(53)=abb113(50)
      acd113(54)=abb113(12)
      acd113(55)=spval4k2(iv1)
      acd113(56)=abb113(31)
      acd113(57)=spval5k2(iv1)
      acd113(58)=abb113(29)
      acd113(59)=spvae2k2(iv1)
      acd113(60)=abb113(28)
      acd113(61)=spvae2l3(iv1)
      acd113(62)=abb113(16)
      acd113(63)=spval4e2(iv1)
      acd113(64)=abb113(26)
      acd113(65)=spval5e2(iv1)
      acd113(66)=abb113(24)
      acd113(67)=spval3e2(iv1)
      acd113(68)=abb113(39)
      acd113(69)=acd113(46)*acd113(53)
      acd113(70)=acd113(43)*acd113(51)
      acd113(71)=acd113(41)*acd113(50)
      acd113(72)=-acd113(20)*acd113(16)
      acd113(73)=acd113(21)*acd113(52)
      acd113(74)=acd113(21)*acd113(28)
      acd113(74)=acd113(74)+acd113(29)
      acd113(75)=acd113(4)*acd113(74)
      acd113(76)=acd113(21)*acd113(22)
      acd113(76)=acd113(76)+acd113(23)
      acd113(77)=acd113(2)*acd113(76)
      acd113(69)=acd113(77)+acd113(75)+acd113(73)+acd113(72)+acd113(71)+acd113(&
      &70)-acd113(54)+acd113(69)
      acd113(69)=acd113(49)*acd113(69)
      acd113(70)=acd113(28)*acd113(34)
      acd113(71)=acd113(22)*acd113(33)
      acd113(70)=acd113(35)+acd113(70)+acd113(71)
      acd113(71)=acd113(40)*acd113(70)
      acd113(72)=acd113(28)*acd113(65)
      acd113(73)=acd113(22)*acd113(63)
      acd113(72)=acd113(72)+acd113(73)
      acd113(72)=acd113(13)*acd113(72)
      acd113(73)=acd113(65)*acd113(37)
      acd113(75)=acd113(63)*acd113(36)
      acd113(77)=acd113(67)*acd113(38)
      acd113(78)=acd113(1)*acd113(7)
      acd113(79)=2.0_ki*acd113(9)
      acd113(80)=acd113(12)*acd113(79)
      acd113(71)=acd113(72)+acd113(71)+acd113(80)+acd113(78)+acd113(77)+acd113(&
      &73)+acd113(75)
      acd113(71)=acd113(6)*acd113(71)
      acd113(72)=acd113(4)*acd113(28)
      acd113(73)=acd113(2)*acd113(22)
      acd113(72)=acd113(73)+acd113(72)+acd113(52)
      acd113(72)=acd113(59)*acd113(72)
      acd113(73)=acd113(27)*acd113(74)
      acd113(74)=acd113(61)*acd113(53)
      acd113(75)=acd113(57)*acd113(51)
      acd113(77)=acd113(55)*acd113(50)
      acd113(76)=acd113(18)*acd113(76)
      acd113(78)=-acd113(16)*acd113(79)
      acd113(72)=acd113(78)+acd113(76)+acd113(77)+acd113(74)+acd113(75)+acd113(&
      &72)+acd113(73)
      acd113(72)=acd113(15)*acd113(72)
      acd113(70)=acd113(32)*acd113(70)
      acd113(73)=acd113(61)*acd113(47)
      acd113(74)=acd113(57)*acd113(44)
      acd113(75)=acd113(55)*acd113(42)
      acd113(76)=acd113(59)*acd113(45)
      acd113(77)=-acd113(14)*acd113(79)
      acd113(70)=acd113(70)+acd113(77)+acd113(76)+acd113(75)+acd113(73)+acd113(&
      &74)
      acd113(70)=acd113(13)*acd113(70)
      acd113(73)=acd113(46)*acd113(47)
      acd113(74)=acd113(43)*acd113(44)
      acd113(75)=acd113(41)*acd113(42)
      acd113(76)=-acd113(20)*acd113(14)
      acd113(77)=acd113(21)*acd113(45)
      acd113(73)=acd113(77)+acd113(76)+acd113(75)+acd113(74)-acd113(48)+acd113(&
      &73)
      acd113(73)=acd113(40)*acd113(73)
      acd113(74)=acd113(34)*acd113(37)
      acd113(75)=acd113(33)*acd113(36)
      acd113(76)=acd113(24)*acd113(38)
      acd113(77)=acd113(19)*acd113(7)
      acd113(78)=acd113(20)*acd113(12)
      acd113(74)=acd113(78)+acd113(77)+acd113(76)+acd113(75)-acd113(39)+acd113(&
      &74)
      acd113(74)=acd113(32)*acd113(74)
      acd113(75)=acd113(24)*acd113(30)
      acd113(76)=acd113(19)*acd113(5)
      acd113(77)=-acd113(20)*acd113(11)
      acd113(75)=acd113(77)+acd113(76)-acd113(31)+acd113(75)
      acd113(75)=acd113(27)*acd113(75)
      acd113(76)=acd113(24)*acd113(25)
      acd113(77)=acd113(19)*acd113(3)
      acd113(78)=-acd113(20)*acd113(10)
      acd113(76)=acd113(78)+acd113(77)-acd113(26)+acd113(76)
      acd113(76)=acd113(18)*acd113(76)
      acd113(77)=acd113(67)*acd113(30)
      acd113(78)=acd113(1)*acd113(5)
      acd113(80)=-acd113(11)*acd113(79)
      acd113(77)=acd113(80)+acd113(77)+acd113(78)
      acd113(77)=acd113(4)*acd113(77)
      acd113(78)=acd113(67)*acd113(25)
      acd113(80)=acd113(1)*acd113(3)
      acd113(81)=-acd113(10)*acd113(79)
      acd113(78)=acd113(81)+acd113(78)+acd113(80)
      acd113(78)=acd113(2)*acd113(78)
      acd113(80)=-acd113(65)*acd113(66)
      acd113(81)=-acd113(63)*acd113(64)
      acd113(82)=-acd113(61)*acd113(62)
      acd113(83)=-acd113(57)*acd113(58)
      acd113(84)=-acd113(55)*acd113(56)
      acd113(85)=-acd113(67)*acd113(68)
      acd113(86)=-acd113(1)*acd113(8)
      acd113(87)=-acd113(59)*acd113(60)
      acd113(79)=acd113(17)*acd113(79)
      brack=acd113(69)+acd113(70)+acd113(71)+acd113(72)+acd113(73)+acd113(74)+a&
      &cd113(75)+acd113(76)+acd113(77)+acd113(78)+acd113(79)+acd113(80)+acd113(&
      &81)+acd113(82)+acd113(83)+acd113(84)+acd113(85)+acd113(86)+acd113(87)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd113h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(88) :: acd113
      complex(ki) :: brack
      acd113(1)=d(iv1,iv2)
      acd113(2)=dotproduct(qshift,spvae1k2)
      acd113(3)=abb113(38)
      acd113(4)=dotproduct(qshift,spval4e1)
      acd113(5)=abb113(23)
      acd113(6)=dotproduct(qshift,spval5e1)
      acd113(7)=abb113(45)
      acd113(8)=dotproduct(qshift,spvae1e2)
      acd113(9)=abb113(20)
      acd113(10)=dotproduct(qshift,spvae2e1)
      acd113(11)=abb113(33)
      acd113(12)=abb113(19)
      acd113(13)=k2(iv1)
      acd113(14)=spval4e1(iv2)
      acd113(15)=abb113(48)
      acd113(16)=spval5e1(iv2)
      acd113(17)=abb113(44)
      acd113(18)=spvae2e1(iv2)
      acd113(19)=abb113(37)
      acd113(20)=k2(iv2)
      acd113(21)=spval4e1(iv1)
      acd113(22)=spval5e1(iv1)
      acd113(23)=spvae2e1(iv1)
      acd113(24)=qshift(iv1)
      acd113(25)=spvae1k2(iv2)
      acd113(26)=spvae1e2(iv2)
      acd113(27)=qshift(iv2)
      acd113(28)=spvae1k2(iv1)
      acd113(29)=spvae1e2(iv1)
      acd113(30)=dotproduct(qshift,spval4e2)
      acd113(31)=abb113(47)
      acd113(32)=dotproduct(qshift,spval5e2)
      acd113(33)=abb113(42)
      acd113(34)=abb113(22)
      acd113(35)=spval4k2(iv2)
      acd113(36)=abb113(10)
      acd113(37)=spval5k2(iv2)
      acd113(38)=abb113(9)
      acd113(39)=spvae2k2(iv2)
      acd113(40)=abb113(34)
      acd113(41)=spvae2l3(iv2)
      acd113(42)=abb113(32)
      acd113(43)=spval4e2(iv2)
      acd113(44)=spval5e2(iv2)
      acd113(45)=spval4k2(iv1)
      acd113(46)=spval5k2(iv1)
      acd113(47)=spvae2k2(iv1)
      acd113(48)=spvae2l3(iv1)
      acd113(49)=spval4e2(iv1)
      acd113(50)=spval5e2(iv1)
      acd113(51)=dotproduct(qshift,spvae2k2)
      acd113(52)=abb113(49)
      acd113(53)=spval3e2(iv2)
      acd113(54)=abb113(11)
      acd113(55)=spval3e2(iv1)
      acd113(56)=abb113(46)
      acd113(57)=abb113(13)
      acd113(58)=abb113(35)
      acd113(59)=abb113(18)
      acd113(60)=abb113(21)
      acd113(61)=abb113(50)
      acd113(62)=abb113(15)
      acd113(63)=abb113(43)
      acd113(64)=abb113(30)
      acd113(65)=-acd113(2)*acd113(1)
      acd113(66)=-acd113(24)*acd113(25)
      acd113(67)=-acd113(27)*acd113(28)
      acd113(65)=acd113(67)+acd113(65)+acd113(66)
      acd113(65)=acd113(3)*acd113(65)
      acd113(66)=-acd113(24)*acd113(14)
      acd113(67)=-acd113(27)*acd113(21)
      acd113(68)=-acd113(4)*acd113(1)
      acd113(66)=acd113(68)+acd113(66)+acd113(67)
      acd113(66)=acd113(5)*acd113(66)
      acd113(67)=-acd113(24)*acd113(16)
      acd113(68)=-acd113(27)*acd113(22)
      acd113(69)=-acd113(6)*acd113(1)
      acd113(67)=acd113(69)+acd113(67)+acd113(68)
      acd113(67)=acd113(7)*acd113(67)
      acd113(68)=-acd113(8)*acd113(1)
      acd113(69)=-acd113(24)*acd113(26)
      acd113(70)=-acd113(27)*acd113(29)
      acd113(68)=acd113(70)+acd113(68)+acd113(69)
      acd113(68)=acd113(9)*acd113(68)
      acd113(69)=acd113(10)*acd113(1)
      acd113(70)=acd113(24)*acd113(18)
      acd113(71)=acd113(27)*acd113(23)
      acd113(69)=acd113(71)+acd113(69)+acd113(70)
      acd113(69)=acd113(11)*acd113(69)
      acd113(70)=acd113(12)*acd113(1)
      acd113(65)=acd113(65)+acd113(66)+acd113(67)+acd113(68)+acd113(69)+acd113(&
      &70)
      acd113(66)=acd113(39)*acd113(29)
      acd113(67)=acd113(47)*acd113(26)
      acd113(66)=acd113(66)+acd113(67)
      acd113(67)=acd113(4)*acd113(66)
      acd113(68)=acd113(2)*acd113(23)
      acd113(69)=acd113(10)*acd113(28)
      acd113(68)=acd113(68)+acd113(69)
      acd113(69)=acd113(43)*acd113(68)
      acd113(70)=acd113(2)*acd113(18)
      acd113(71)=acd113(10)*acd113(25)
      acd113(70)=acd113(70)+acd113(71)
      acd113(71)=acd113(49)*acd113(70)
      acd113(72)=acd113(25)*acd113(23)
      acd113(73)=acd113(28)*acd113(18)
      acd113(72)=acd113(72)+acd113(73)
      acd113(73)=acd113(30)*acd113(72)
      acd113(67)=acd113(73)+acd113(71)+acd113(69)+acd113(67)
      acd113(67)=acd113(31)*acd113(67)
      acd113(69)=acd113(6)*acd113(66)
      acd113(68)=acd113(44)*acd113(68)
      acd113(70)=acd113(50)*acd113(70)
      acd113(71)=acd113(32)*acd113(72)
      acd113(68)=acd113(71)+acd113(70)+acd113(68)+acd113(69)
      acd113(68)=acd113(33)*acd113(68)
      acd113(69)=acd113(45)*acd113(36)
      acd113(70)=acd113(46)*acd113(38)
      acd113(71)=acd113(48)*acd113(42)
      acd113(69)=acd113(71)+acd113(70)+acd113(69)
      acd113(69)=acd113(25)*acd113(69)
      acd113(70)=acd113(36)*acd113(35)
      acd113(71)=acd113(38)*acd113(37)
      acd113(73)=acd113(42)*acd113(41)
      acd113(70)=acd113(73)+acd113(71)+acd113(70)
      acd113(70)=acd113(28)*acd113(70)
      acd113(71)=acd113(21)*acd113(31)
      acd113(73)=acd113(22)*acd113(33)
      acd113(71)=acd113(71)+acd113(73)
      acd113(73)=acd113(39)*acd113(71)
      acd113(74)=acd113(14)*acd113(31)
      acd113(75)=acd113(16)*acd113(33)
      acd113(74)=acd113(74)+acd113(75)
      acd113(75)=acd113(47)*acd113(74)
      acd113(73)=acd113(73)+acd113(75)
      acd113(73)=acd113(8)*acd113(73)
      acd113(71)=acd113(26)*acd113(71)
      acd113(74)=acd113(29)*acd113(74)
      acd113(71)=acd113(74)+acd113(71)
      acd113(71)=acd113(51)*acd113(71)
      acd113(74)=acd113(13)*acd113(14)
      acd113(75)=acd113(20)*acd113(21)
      acd113(74)=acd113(74)+acd113(75)
      acd113(74)=acd113(15)*acd113(74)
      acd113(75)=acd113(13)*acd113(16)
      acd113(76)=acd113(20)*acd113(22)
      acd113(75)=acd113(75)+acd113(76)
      acd113(75)=acd113(17)*acd113(75)
      acd113(76)=acd113(13)*acd113(18)
      acd113(77)=acd113(20)*acd113(23)
      acd113(76)=acd113(76)+acd113(77)
      acd113(76)=acd113(19)*acd113(76)
      acd113(72)=acd113(34)*acd113(72)
      acd113(77)=acd113(39)*acd113(28)
      acd113(78)=acd113(47)*acd113(25)
      acd113(77)=acd113(77)+acd113(78)
      acd113(77)=acd113(40)*acd113(77)
      acd113(78)=acd113(14)*acd113(29)
      acd113(79)=acd113(21)*acd113(26)
      acd113(78)=acd113(78)+acd113(79)
      acd113(78)=acd113(52)*acd113(78)
      acd113(79)=acd113(53)*acd113(21)
      acd113(80)=acd113(55)*acd113(14)
      acd113(79)=acd113(79)+acd113(80)
      acd113(79)=acd113(54)*acd113(79)
      acd113(80)=acd113(16)*acd113(29)
      acd113(81)=acd113(22)*acd113(26)
      acd113(80)=acd113(80)+acd113(81)
      acd113(80)=acd113(56)*acd113(80)
      acd113(81)=acd113(53)*acd113(22)
      acd113(82)=acd113(55)*acd113(16)
      acd113(81)=acd113(81)+acd113(82)
      acd113(81)=acd113(57)*acd113(81)
      acd113(82)=acd113(35)*acd113(29)
      acd113(83)=acd113(45)*acd113(26)
      acd113(82)=acd113(82)+acd113(83)
      acd113(82)=acd113(58)*acd113(82)
      acd113(83)=acd113(37)*acd113(29)
      acd113(84)=acd113(46)*acd113(26)
      acd113(83)=acd113(83)+acd113(84)
      acd113(83)=acd113(59)*acd113(83)
      acd113(66)=acd113(60)*acd113(66)
      acd113(84)=acd113(41)*acd113(29)
      acd113(85)=acd113(48)*acd113(26)
      acd113(84)=acd113(84)+acd113(85)
      acd113(84)=acd113(61)*acd113(84)
      acd113(85)=acd113(43)*acd113(23)
      acd113(86)=acd113(49)*acd113(18)
      acd113(85)=acd113(85)+acd113(86)
      acd113(85)=acd113(62)*acd113(85)
      acd113(86)=acd113(44)*acd113(23)
      acd113(87)=acd113(50)*acd113(18)
      acd113(86)=acd113(86)+acd113(87)
      acd113(86)=acd113(63)*acd113(86)
      acd113(87)=acd113(53)*acd113(23)
      acd113(88)=acd113(55)*acd113(18)
      acd113(87)=acd113(87)+acd113(88)
      acd113(87)=acd113(64)*acd113(87)
      brack=2.0_ki*acd113(65)+acd113(66)+acd113(67)+acd113(68)+acd113(69)+acd11&
      &3(70)+acd113(71)+acd113(72)+acd113(73)+acd113(74)+acd113(75)+acd113(76)+&
      &acd113(77)+acd113(78)+acd113(79)+acd113(80)+acd113(81)+acd113(82)+acd113&
      &(83)+acd113(84)+acd113(85)+acd113(86)+acd113(87)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd113h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(45) :: acd113
      complex(ki) :: brack
      acd113(1)=d(iv1,iv2)
      acd113(2)=spvae1k2(iv3)
      acd113(3)=abb113(38)
      acd113(4)=spval4e1(iv3)
      acd113(5)=abb113(23)
      acd113(6)=spval5e1(iv3)
      acd113(7)=abb113(45)
      acd113(8)=spvae1e2(iv3)
      acd113(9)=abb113(20)
      acd113(10)=spvae2e1(iv3)
      acd113(11)=abb113(33)
      acd113(12)=d(iv1,iv3)
      acd113(13)=spvae1k2(iv2)
      acd113(14)=spval4e1(iv2)
      acd113(15)=spval5e1(iv2)
      acd113(16)=spvae1e2(iv2)
      acd113(17)=spvae2e1(iv2)
      acd113(18)=d(iv2,iv3)
      acd113(19)=spvae1k2(iv1)
      acd113(20)=spval4e1(iv1)
      acd113(21)=spval5e1(iv1)
      acd113(22)=spvae1e2(iv1)
      acd113(23)=spvae2e1(iv1)
      acd113(24)=spval4e2(iv3)
      acd113(25)=abb113(47)
      acd113(26)=spval5e2(iv3)
      acd113(27)=abb113(42)
      acd113(28)=spval4e2(iv2)
      acd113(29)=spval5e2(iv2)
      acd113(30)=spval4e2(iv1)
      acd113(31)=spval5e2(iv1)
      acd113(32)=spvae2k2(iv3)
      acd113(33)=spvae2k2(iv2)
      acd113(34)=spvae2k2(iv1)
      acd113(35)=-acd113(2)*acd113(1)
      acd113(36)=-acd113(13)*acd113(12)
      acd113(37)=-acd113(19)*acd113(18)
      acd113(35)=acd113(37)+acd113(35)+acd113(36)
      acd113(35)=acd113(3)*acd113(35)
      acd113(36)=-acd113(8)*acd113(1)
      acd113(37)=-acd113(16)*acd113(12)
      acd113(38)=-acd113(22)*acd113(18)
      acd113(36)=acd113(38)+acd113(36)+acd113(37)
      acd113(36)=acd113(9)*acd113(36)
      acd113(37)=acd113(10)*acd113(1)
      acd113(38)=acd113(17)*acd113(12)
      acd113(39)=acd113(23)*acd113(18)
      acd113(37)=acd113(39)+acd113(37)+acd113(38)
      acd113(37)=acd113(11)*acd113(37)
      acd113(35)=acd113(37)+acd113(35)+acd113(36)
      acd113(36)=acd113(19)*acd113(17)
      acd113(37)=acd113(23)*acd113(13)
      acd113(36)=acd113(36)+acd113(37)
      acd113(37)=acd113(24)*acd113(36)
      acd113(38)=acd113(19)*acd113(10)
      acd113(39)=acd113(23)*acd113(2)
      acd113(38)=acd113(38)+acd113(39)
      acd113(39)=acd113(28)*acd113(38)
      acd113(40)=acd113(13)*acd113(10)
      acd113(41)=acd113(17)*acd113(2)
      acd113(40)=acd113(40)+acd113(41)
      acd113(41)=acd113(30)*acd113(40)
      acd113(37)=acd113(41)+acd113(39)+acd113(37)
      acd113(37)=acd113(25)*acd113(37)
      acd113(36)=acd113(26)*acd113(36)
      acd113(38)=acd113(29)*acd113(38)
      acd113(39)=acd113(31)*acd113(40)
      acd113(36)=acd113(39)+acd113(38)+acd113(36)
      acd113(36)=acd113(27)*acd113(36)
      acd113(38)=acd113(33)*acd113(22)
      acd113(39)=acd113(34)*acd113(16)
      acd113(38)=acd113(38)+acd113(39)
      acd113(39)=acd113(25)*acd113(38)
      acd113(40)=2.0_ki*acd113(5)
      acd113(41)=-acd113(1)*acd113(40)
      acd113(39)=acd113(41)+acd113(39)
      acd113(39)=acd113(4)*acd113(39)
      acd113(38)=acd113(27)*acd113(38)
      acd113(41)=2.0_ki*acd113(7)
      acd113(42)=-acd113(1)*acd113(41)
      acd113(38)=acd113(42)+acd113(38)
      acd113(38)=acd113(6)*acd113(38)
      acd113(42)=acd113(32)*acd113(22)
      acd113(43)=acd113(34)*acd113(8)
      acd113(42)=acd113(42)+acd113(43)
      acd113(43)=acd113(25)*acd113(42)
      acd113(44)=-acd113(12)*acd113(40)
      acd113(43)=acd113(44)+acd113(43)
      acd113(43)=acd113(14)*acd113(43)
      acd113(42)=acd113(27)*acd113(42)
      acd113(44)=-acd113(12)*acd113(41)
      acd113(42)=acd113(44)+acd113(42)
      acd113(42)=acd113(15)*acd113(42)
      acd113(44)=acd113(32)*acd113(16)
      acd113(45)=acd113(33)*acd113(8)
      acd113(44)=acd113(44)+acd113(45)
      acd113(45)=acd113(25)*acd113(44)
      acd113(40)=-acd113(18)*acd113(40)
      acd113(40)=acd113(40)+acd113(45)
      acd113(40)=acd113(20)*acd113(40)
      acd113(44)=acd113(27)*acd113(44)
      acd113(41)=-acd113(18)*acd113(41)
      acd113(41)=acd113(41)+acd113(44)
      acd113(41)=acd113(21)*acd113(41)
      brack=2.0_ki*acd113(35)+acd113(36)+acd113(37)+acd113(38)+acd113(39)+acd11&
      &3(40)+acd113(41)+acd113(42)+acd113(43)
   end function brack_4
!---#] function brack_4:
!---#[ function brack_5:
   pure function brack_5(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd113h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd113
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_5
!---#] function brack_5:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3,i4) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd113h0
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
      qshift = -k2+k3+k5
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
end module     p2_gg_httbar_d113h0l1d
