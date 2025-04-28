module     p2_gg_httbar_d101h12l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d101h12l1d.f90
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
      use p2_gg_httbar_abbrevd101h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(70) :: acd101
      complex(ki) :: brack
      acd101(1)=dotproduct(qshift,qshift)
      acd101(2)=dotproduct(qshift,spvak2e2)
      acd101(3)=abb101(24)
      acd101(4)=dotproduct(qshift,spvae2l4)
      acd101(5)=abb101(18)
      acd101(6)=dotproduct(qshift,spvae2l5)
      acd101(7)=abb101(40)
      acd101(8)=dotproduct(qshift,spvae1e2)
      acd101(9)=abb101(36)
      acd101(10)=dotproduct(qshift,spvae2e1)
      acd101(11)=abb101(30)
      acd101(12)=abb101(13)
      acd101(13)=dotproduct(qshift,spvae1l4)
      acd101(14)=abb101(46)
      acd101(15)=dotproduct(qshift,spvae1l5)
      acd101(16)=abb101(45)
      acd101(17)=abb101(43)
      acd101(18)=dotproduct(qshift,spvak1l4)
      acd101(19)=abb101(7)
      acd101(20)=dotproduct(qshift,spvak1l5)
      acd101(21)=abb101(8)
      acd101(22)=dotproduct(qshift,spvak1e1)
      acd101(23)=abb101(39)
      acd101(24)=dotproduct(qshift,spvak2e1)
      acd101(25)=abb101(16)
      acd101(26)=dotproduct(qshift,spval3e1)
      acd101(27)=abb101(33)
      acd101(28)=abb101(9)
      acd101(29)=abb101(23)
      acd101(30)=dotproduct(qshift,spvak2k1)
      acd101(31)=abb101(44)
      acd101(32)=dotproduct(qshift,spvae1k1)
      acd101(33)=abb101(47)
      acd101(34)=dotproduct(qshift,spvae1l3)
      acd101(35)=abb101(31)
      acd101(36)=abb101(14)
      acd101(37)=abb101(21)
      acd101(38)=abb101(37)
      acd101(39)=abb101(48)
      acd101(40)=abb101(41)
      acd101(41)=abb101(10)
      acd101(42)=abb101(38)
      acd101(43)=abb101(49)
      acd101(44)=abb101(35)
      acd101(45)=abb101(34)
      acd101(46)=abb101(42)
      acd101(47)=abb101(26)
      acd101(48)=abb101(29)
      acd101(49)=abb101(27)
      acd101(50)=abb101(22)
      acd101(51)=abb101(19)
      acd101(52)=abb101(15)
      acd101(53)=abb101(11)
      acd101(54)=abb101(25)
      acd101(55)=abb101(20)
      acd101(56)=abb101(17)
      acd101(57)=abb101(32)
      acd101(58)=abb101(12)
      acd101(59)=acd101(26)*acd101(46)
      acd101(60)=acd101(22)*acd101(44)
      acd101(61)=acd101(20)*acd101(43)
      acd101(62)=acd101(18)*acd101(42)
      acd101(63)=-acd101(24)*acd101(45)
      acd101(64)=-acd101(1)*acd101(9)
      acd101(65)=acd101(24)*acd101(16)
      acd101(65)=acd101(37)+acd101(65)
      acd101(65)=acd101(6)*acd101(65)
      acd101(66)=acd101(24)*acd101(14)
      acd101(66)=acd101(29)+acd101(66)
      acd101(66)=acd101(4)*acd101(66)
      acd101(59)=acd101(66)+acd101(65)+acd101(64)+acd101(63)+acd101(62)+acd101(&
      &61)+acd101(60)-acd101(47)+acd101(59)
      acd101(59)=acd101(8)*acd101(59)
      acd101(60)=acd101(15)*acd101(16)
      acd101(61)=acd101(13)*acd101(14)
      acd101(60)=acd101(61)+acd101(17)+acd101(60)
      acd101(60)=acd101(10)*acd101(60)
      acd101(61)=acd101(26)*acd101(27)
      acd101(62)=acd101(22)*acd101(23)
      acd101(63)=acd101(20)*acd101(21)
      acd101(64)=acd101(18)*acd101(19)
      acd101(65)=acd101(24)*acd101(25)
      acd101(66)=-acd101(1)*acd101(3)
      acd101(60)=acd101(60)+acd101(66)+acd101(65)+acd101(64)+acd101(63)+acd101(&
      &62)-acd101(28)+acd101(61)
      acd101(60)=acd101(2)*acd101(60)
      acd101(61)=acd101(34)*acd101(40)
      acd101(62)=acd101(32)*acd101(39)
      acd101(63)=acd101(30)*acd101(38)
      acd101(64)=-acd101(1)*acd101(7)
      acd101(61)=acd101(64)+acd101(63)+acd101(62)-acd101(41)+acd101(61)
      acd101(61)=acd101(6)*acd101(61)
      acd101(62)=acd101(34)*acd101(35)
      acd101(63)=acd101(32)*acd101(33)
      acd101(64)=acd101(30)*acd101(31)
      acd101(65)=-acd101(1)*acd101(5)
      acd101(62)=acd101(65)+acd101(64)+acd101(63)-acd101(36)+acd101(62)
      acd101(62)=acd101(4)*acd101(62)
      acd101(63)=acd101(34)*acd101(50)
      acd101(64)=acd101(32)*acd101(49)
      acd101(65)=acd101(30)*acd101(48)
      acd101(66)=-acd101(1)*acd101(11)
      acd101(63)=acd101(66)+acd101(65)+acd101(64)-acd101(51)+acd101(63)
      acd101(63)=acd101(10)*acd101(63)
      acd101(64)=-acd101(15)*acd101(56)
      acd101(65)=-acd101(13)*acd101(55)
      acd101(66)=-acd101(34)*acd101(57)
      acd101(67)=-acd101(32)*acd101(53)
      acd101(68)=-acd101(30)*acd101(52)
      acd101(69)=-acd101(24)*acd101(54)
      acd101(70)=acd101(1)*acd101(12)
      brack=acd101(58)+acd101(59)+acd101(60)+acd101(61)+acd101(62)+acd101(63)+a&
      &cd101(64)+acd101(65)+acd101(66)+acd101(67)+acd101(68)+acd101(69)+acd101(&
      &70)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd101h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(90) :: acd101
      complex(ki) :: brack
      acd101(1)=qshift(iv1)
      acd101(2)=dotproduct(qshift,spvak2e2)
      acd101(3)=abb101(24)
      acd101(4)=dotproduct(qshift,spvae2l4)
      acd101(5)=abb101(18)
      acd101(6)=dotproduct(qshift,spvae2l5)
      acd101(7)=abb101(40)
      acd101(8)=dotproduct(qshift,spvae1e2)
      acd101(9)=abb101(36)
      acd101(10)=dotproduct(qshift,spvae2e1)
      acd101(11)=abb101(30)
      acd101(12)=abb101(13)
      acd101(13)=spvak2e2(iv1)
      acd101(14)=dotproduct(qshift,qshift)
      acd101(15)=dotproduct(qshift,spvae1l4)
      acd101(16)=abb101(46)
      acd101(17)=dotproduct(qshift,spvae1l5)
      acd101(18)=abb101(45)
      acd101(19)=abb101(43)
      acd101(20)=dotproduct(qshift,spvak1l4)
      acd101(21)=abb101(7)
      acd101(22)=dotproduct(qshift,spvak1l5)
      acd101(23)=abb101(8)
      acd101(24)=dotproduct(qshift,spvak1e1)
      acd101(25)=abb101(39)
      acd101(26)=dotproduct(qshift,spvak2e1)
      acd101(27)=abb101(16)
      acd101(28)=dotproduct(qshift,spval3e1)
      acd101(29)=abb101(33)
      acd101(30)=abb101(9)
      acd101(31)=spvae2l4(iv1)
      acd101(32)=abb101(23)
      acd101(33)=dotproduct(qshift,spvak2k1)
      acd101(34)=abb101(44)
      acd101(35)=dotproduct(qshift,spvae1k1)
      acd101(36)=abb101(47)
      acd101(37)=dotproduct(qshift,spvae1l3)
      acd101(38)=abb101(31)
      acd101(39)=abb101(14)
      acd101(40)=spvae2l5(iv1)
      acd101(41)=abb101(21)
      acd101(42)=abb101(37)
      acd101(43)=abb101(48)
      acd101(44)=abb101(41)
      acd101(45)=abb101(10)
      acd101(46)=spvae1e2(iv1)
      acd101(47)=abb101(38)
      acd101(48)=abb101(49)
      acd101(49)=abb101(35)
      acd101(50)=abb101(34)
      acd101(51)=abb101(42)
      acd101(52)=abb101(26)
      acd101(53)=spvae2e1(iv1)
      acd101(54)=abb101(29)
      acd101(55)=abb101(27)
      acd101(56)=abb101(22)
      acd101(57)=abb101(19)
      acd101(58)=spvak1l4(iv1)
      acd101(59)=spvak1l5(iv1)
      acd101(60)=spvak2k1(iv1)
      acd101(61)=abb101(15)
      acd101(62)=spvak1e1(iv1)
      acd101(63)=spvae1k1(iv1)
      acd101(64)=abb101(11)
      acd101(65)=spvak2e1(iv1)
      acd101(66)=abb101(25)
      acd101(67)=spval3e1(iv1)
      acd101(68)=spvae1l4(iv1)
      acd101(69)=abb101(20)
      acd101(70)=spvae1l5(iv1)
      acd101(71)=abb101(17)
      acd101(72)=spvae1l3(iv1)
      acd101(73)=abb101(32)
      acd101(74)=acd101(28)*acd101(51)
      acd101(75)=acd101(24)*acd101(49)
      acd101(76)=acd101(22)*acd101(48)
      acd101(77)=acd101(20)*acd101(47)
      acd101(78)=-acd101(14)*acd101(9)
      acd101(79)=-acd101(26)*acd101(50)
      acd101(80)=acd101(18)*acd101(26)
      acd101(80)=acd101(80)+acd101(41)
      acd101(81)=acd101(6)*acd101(80)
      acd101(82)=acd101(16)*acd101(26)
      acd101(82)=acd101(82)+acd101(32)
      acd101(83)=acd101(4)*acd101(82)
      acd101(74)=acd101(83)+acd101(81)+acd101(79)+acd101(78)+acd101(77)+acd101(&
      &76)+acd101(75)-acd101(52)+acd101(74)
      acd101(74)=acd101(46)*acd101(74)
      acd101(75)=acd101(18)*acd101(17)
      acd101(76)=acd101(16)*acd101(15)
      acd101(75)=acd101(19)+acd101(75)+acd101(76)
      acd101(76)=acd101(53)*acd101(75)
      acd101(77)=acd101(18)*acd101(70)
      acd101(78)=acd101(16)*acd101(68)
      acd101(77)=acd101(77)+acd101(78)
      acd101(77)=acd101(10)*acd101(77)
      acd101(78)=acd101(29)*acd101(67)
      acd101(79)=acd101(25)*acd101(62)
      acd101(81)=acd101(23)*acd101(59)
      acd101(83)=acd101(21)*acd101(58)
      acd101(84)=acd101(65)*acd101(27)
      acd101(85)=2.0_ki*acd101(1)
      acd101(86)=-acd101(3)*acd101(85)
      acd101(76)=acd101(77)+acd101(76)+acd101(86)+acd101(84)+acd101(83)+acd101(&
      &81)+acd101(78)+acd101(79)
      acd101(76)=acd101(2)*acd101(76)
      acd101(77)=acd101(6)*acd101(18)
      acd101(78)=acd101(4)*acd101(16)
      acd101(77)=acd101(78)+acd101(77)-acd101(50)
      acd101(77)=acd101(65)*acd101(77)
      acd101(78)=acd101(51)*acd101(67)
      acd101(79)=acd101(49)*acd101(62)
      acd101(81)=acd101(48)*acd101(59)
      acd101(83)=acd101(47)*acd101(58)
      acd101(84)=-acd101(9)*acd101(85)
      acd101(80)=acd101(40)*acd101(80)
      acd101(82)=acd101(31)*acd101(82)
      acd101(77)=acd101(82)+acd101(80)+acd101(84)+acd101(83)+acd101(81)+acd101(&
      &78)+acd101(79)+acd101(77)
      acd101(77)=acd101(8)*acd101(77)
      acd101(75)=acd101(10)*acd101(75)
      acd101(78)=acd101(28)*acd101(29)
      acd101(79)=acd101(24)*acd101(25)
      acd101(80)=acd101(22)*acd101(23)
      acd101(81)=acd101(20)*acd101(21)
      acd101(82)=-acd101(14)*acd101(3)
      acd101(83)=acd101(26)*acd101(27)
      acd101(75)=acd101(75)+acd101(83)+acd101(82)+acd101(81)+acd101(80)+acd101(&
      &79)-acd101(30)+acd101(78)
      acd101(75)=acd101(13)*acd101(75)
      acd101(78)=acd101(37)*acd101(44)
      acd101(79)=acd101(35)*acd101(43)
      acd101(80)=acd101(33)*acd101(42)
      acd101(81)=-acd101(14)*acd101(7)
      acd101(78)=acd101(81)+acd101(80)+acd101(79)-acd101(45)+acd101(78)
      acd101(78)=acd101(40)*acd101(78)
      acd101(79)=acd101(37)*acd101(38)
      acd101(80)=acd101(35)*acd101(36)
      acd101(81)=acd101(33)*acd101(34)
      acd101(82)=-acd101(14)*acd101(5)
      acd101(79)=acd101(82)+acd101(81)+acd101(80)-acd101(39)+acd101(79)
      acd101(79)=acd101(31)*acd101(79)
      acd101(80)=acd101(72)*acd101(44)
      acd101(81)=acd101(63)*acd101(43)
      acd101(82)=acd101(60)*acd101(42)
      acd101(83)=-acd101(7)*acd101(85)
      acd101(80)=acd101(83)+acd101(82)+acd101(80)+acd101(81)
      acd101(80)=acd101(6)*acd101(80)
      acd101(81)=acd101(72)*acd101(38)
      acd101(82)=acd101(63)*acd101(36)
      acd101(83)=acd101(60)*acd101(34)
      acd101(84)=-acd101(5)*acd101(85)
      acd101(81)=acd101(84)+acd101(83)+acd101(81)+acd101(82)
      acd101(81)=acd101(4)*acd101(81)
      acd101(82)=acd101(37)*acd101(56)
      acd101(83)=acd101(35)*acd101(55)
      acd101(84)=acd101(33)*acd101(54)
      acd101(86)=-acd101(14)*acd101(11)
      acd101(82)=acd101(86)+acd101(84)+acd101(83)-acd101(57)+acd101(82)
      acd101(82)=acd101(53)*acd101(82)
      acd101(83)=acd101(72)*acd101(56)
      acd101(84)=acd101(63)*acd101(55)
      acd101(86)=acd101(60)*acd101(54)
      acd101(87)=-acd101(11)*acd101(85)
      acd101(83)=acd101(87)+acd101(86)+acd101(83)+acd101(84)
      acd101(83)=acd101(10)*acd101(83)
      acd101(84)=-acd101(70)*acd101(71)
      acd101(86)=-acd101(68)*acd101(69)
      acd101(87)=-acd101(72)*acd101(73)
      acd101(88)=-acd101(63)*acd101(64)
      acd101(89)=-acd101(60)*acd101(61)
      acd101(90)=-acd101(65)*acd101(66)
      acd101(85)=acd101(12)*acd101(85)
      brack=acd101(74)+acd101(75)+acd101(76)+acd101(77)+acd101(78)+acd101(79)+a&
      &cd101(80)+acd101(81)+acd101(82)+acd101(83)+acd101(84)+acd101(85)+acd101(&
      &86)+acd101(87)+acd101(88)+acd101(89)+acd101(90)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd101h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(97) :: acd101
      complex(ki) :: brack
      acd101(1)=d(iv1,iv2)
      acd101(2)=dotproduct(qshift,spvak2e2)
      acd101(3)=abb101(24)
      acd101(4)=dotproduct(qshift,spvae2l4)
      acd101(5)=abb101(18)
      acd101(6)=dotproduct(qshift,spvae2l5)
      acd101(7)=abb101(40)
      acd101(8)=dotproduct(qshift,spvae1e2)
      acd101(9)=abb101(36)
      acd101(10)=dotproduct(qshift,spvae2e1)
      acd101(11)=abb101(30)
      acd101(12)=abb101(13)
      acd101(13)=qshift(iv1)
      acd101(14)=spvak2e2(iv2)
      acd101(15)=spvae2l4(iv2)
      acd101(16)=spvae2l5(iv2)
      acd101(17)=spvae1e2(iv2)
      acd101(18)=spvae2e1(iv2)
      acd101(19)=qshift(iv2)
      acd101(20)=spvak2e2(iv1)
      acd101(21)=spvae2l4(iv1)
      acd101(22)=spvae2l5(iv1)
      acd101(23)=spvae1e2(iv1)
      acd101(24)=spvae2e1(iv1)
      acd101(25)=dotproduct(qshift,spvae1l4)
      acd101(26)=abb101(46)
      acd101(27)=dotproduct(qshift,spvae1l5)
      acd101(28)=abb101(45)
      acd101(29)=abb101(43)
      acd101(30)=spvak1l4(iv2)
      acd101(31)=abb101(7)
      acd101(32)=spvak1l5(iv2)
      acd101(33)=abb101(8)
      acd101(34)=spvak1e1(iv2)
      acd101(35)=abb101(39)
      acd101(36)=spvak2e1(iv2)
      acd101(37)=abb101(16)
      acd101(38)=spval3e1(iv2)
      acd101(39)=abb101(33)
      acd101(40)=spvae1l4(iv2)
      acd101(41)=spvae1l5(iv2)
      acd101(42)=spvak1l4(iv1)
      acd101(43)=spvak1l5(iv1)
      acd101(44)=spvak1e1(iv1)
      acd101(45)=spvak2e1(iv1)
      acd101(46)=spval3e1(iv1)
      acd101(47)=spvae1l4(iv1)
      acd101(48)=spvae1l5(iv1)
      acd101(49)=dotproduct(qshift,spvak2e1)
      acd101(50)=abb101(23)
      acd101(51)=spvak2k1(iv2)
      acd101(52)=abb101(44)
      acd101(53)=spvae1k1(iv2)
      acd101(54)=abb101(47)
      acd101(55)=spvae1l3(iv2)
      acd101(56)=abb101(31)
      acd101(57)=spvak2k1(iv1)
      acd101(58)=spvae1k1(iv1)
      acd101(59)=spvae1l3(iv1)
      acd101(60)=abb101(21)
      acd101(61)=abb101(37)
      acd101(62)=abb101(48)
      acd101(63)=abb101(41)
      acd101(64)=abb101(38)
      acd101(65)=abb101(49)
      acd101(66)=abb101(35)
      acd101(67)=abb101(34)
      acd101(68)=abb101(42)
      acd101(69)=abb101(29)
      acd101(70)=abb101(27)
      acd101(71)=abb101(22)
      acd101(72)=-acd101(2)*acd101(1)
      acd101(73)=-acd101(13)*acd101(14)
      acd101(74)=-acd101(19)*acd101(20)
      acd101(72)=acd101(74)+acd101(72)+acd101(73)
      acd101(72)=acd101(3)*acd101(72)
      acd101(73)=-acd101(13)*acd101(15)
      acd101(74)=-acd101(19)*acd101(21)
      acd101(75)=-acd101(4)*acd101(1)
      acd101(73)=acd101(75)+acd101(73)+acd101(74)
      acd101(73)=acd101(5)*acd101(73)
      acd101(74)=-acd101(13)*acd101(16)
      acd101(75)=-acd101(19)*acd101(22)
      acd101(76)=-acd101(6)*acd101(1)
      acd101(74)=acd101(76)+acd101(74)+acd101(75)
      acd101(74)=acd101(7)*acd101(74)
      acd101(75)=-acd101(8)*acd101(1)
      acd101(76)=-acd101(13)*acd101(17)
      acd101(77)=-acd101(19)*acd101(23)
      acd101(75)=acd101(77)+acd101(75)+acd101(76)
      acd101(75)=acd101(9)*acd101(75)
      acd101(76)=-acd101(10)*acd101(1)
      acd101(77)=-acd101(13)*acd101(18)
      acd101(78)=-acd101(19)*acd101(24)
      acd101(76)=acd101(78)+acd101(76)+acd101(77)
      acd101(76)=acd101(11)*acd101(76)
      acd101(77)=acd101(12)*acd101(1)
      acd101(72)=acd101(72)+acd101(73)+acd101(74)+acd101(75)+acd101(76)+acd101(&
      &77)
      acd101(73)=acd101(36)*acd101(23)
      acd101(74)=acd101(45)*acd101(17)
      acd101(73)=acd101(73)+acd101(74)
      acd101(74)=acd101(4)*acd101(73)
      acd101(75)=acd101(18)*acd101(20)
      acd101(76)=acd101(24)*acd101(14)
      acd101(75)=acd101(75)+acd101(76)
      acd101(76)=acd101(25)*acd101(75)
      acd101(77)=acd101(2)*acd101(24)
      acd101(78)=acd101(10)*acd101(20)
      acd101(77)=acd101(77)+acd101(78)
      acd101(78)=acd101(40)*acd101(77)
      acd101(79)=acd101(2)*acd101(18)
      acd101(80)=acd101(10)*acd101(14)
      acd101(79)=acd101(79)+acd101(80)
      acd101(80)=acd101(47)*acd101(79)
      acd101(74)=acd101(80)+acd101(78)+acd101(76)+acd101(74)
      acd101(74)=acd101(26)*acd101(74)
      acd101(76)=acd101(6)*acd101(73)
      acd101(78)=acd101(27)*acd101(75)
      acd101(77)=acd101(41)*acd101(77)
      acd101(79)=acd101(48)*acd101(79)
      acd101(76)=acd101(79)+acd101(77)+acd101(78)+acd101(76)
      acd101(76)=acd101(28)*acd101(76)
      acd101(77)=acd101(42)*acd101(31)
      acd101(78)=acd101(43)*acd101(33)
      acd101(79)=acd101(44)*acd101(35)
      acd101(80)=acd101(46)*acd101(39)
      acd101(77)=acd101(80)+acd101(79)+acd101(78)+acd101(77)
      acd101(77)=acd101(14)*acd101(77)
      acd101(78)=acd101(31)*acd101(30)
      acd101(79)=acd101(33)*acd101(32)
      acd101(80)=acd101(35)*acd101(34)
      acd101(81)=acd101(39)*acd101(38)
      acd101(78)=acd101(81)+acd101(80)+acd101(79)+acd101(78)
      acd101(78)=acd101(20)*acd101(78)
      acd101(79)=acd101(21)*acd101(26)
      acd101(80)=acd101(22)*acd101(28)
      acd101(79)=acd101(79)+acd101(80)
      acd101(79)=acd101(36)*acd101(79)
      acd101(80)=acd101(15)*acd101(26)
      acd101(81)=acd101(16)*acd101(28)
      acd101(80)=acd101(80)+acd101(81)
      acd101(80)=acd101(45)*acd101(80)
      acd101(79)=acd101(79)+acd101(80)
      acd101(79)=acd101(8)*acd101(79)
      acd101(80)=acd101(15)*acd101(23)
      acd101(81)=acd101(21)*acd101(17)
      acd101(80)=acd101(80)+acd101(81)
      acd101(81)=acd101(26)*acd101(80)
      acd101(82)=acd101(16)*acd101(23)
      acd101(83)=acd101(22)*acd101(17)
      acd101(82)=acd101(82)+acd101(83)
      acd101(83)=acd101(28)*acd101(82)
      acd101(81)=acd101(83)+acd101(81)
      acd101(81)=acd101(49)*acd101(81)
      acd101(75)=acd101(29)*acd101(75)
      acd101(83)=acd101(36)*acd101(20)
      acd101(84)=acd101(45)*acd101(14)
      acd101(83)=acd101(83)+acd101(84)
      acd101(83)=acd101(37)*acd101(83)
      acd101(80)=acd101(50)*acd101(80)
      acd101(84)=acd101(51)*acd101(21)
      acd101(85)=acd101(57)*acd101(15)
      acd101(84)=acd101(84)+acd101(85)
      acd101(84)=acd101(52)*acd101(84)
      acd101(85)=acd101(53)*acd101(21)
      acd101(86)=acd101(58)*acd101(15)
      acd101(85)=acd101(85)+acd101(86)
      acd101(85)=acd101(54)*acd101(85)
      acd101(86)=acd101(55)*acd101(21)
      acd101(87)=acd101(59)*acd101(15)
      acd101(86)=acd101(86)+acd101(87)
      acd101(86)=acd101(56)*acd101(86)
      acd101(82)=acd101(60)*acd101(82)
      acd101(87)=acd101(51)*acd101(22)
      acd101(88)=acd101(57)*acd101(16)
      acd101(87)=acd101(87)+acd101(88)
      acd101(87)=acd101(61)*acd101(87)
      acd101(88)=acd101(53)*acd101(22)
      acd101(89)=acd101(58)*acd101(16)
      acd101(88)=acd101(88)+acd101(89)
      acd101(88)=acd101(62)*acd101(88)
      acd101(89)=acd101(55)*acd101(22)
      acd101(90)=acd101(59)*acd101(16)
      acd101(89)=acd101(89)+acd101(90)
      acd101(89)=acd101(63)*acd101(89)
      acd101(90)=acd101(30)*acd101(23)
      acd101(91)=acd101(42)*acd101(17)
      acd101(90)=acd101(90)+acd101(91)
      acd101(90)=acd101(64)*acd101(90)
      acd101(91)=acd101(32)*acd101(23)
      acd101(92)=acd101(43)*acd101(17)
      acd101(91)=acd101(91)+acd101(92)
      acd101(91)=acd101(65)*acd101(91)
      acd101(92)=acd101(34)*acd101(23)
      acd101(93)=acd101(44)*acd101(17)
      acd101(92)=acd101(92)+acd101(93)
      acd101(92)=acd101(66)*acd101(92)
      acd101(73)=-acd101(67)*acd101(73)
      acd101(93)=acd101(38)*acd101(23)
      acd101(94)=acd101(46)*acd101(17)
      acd101(93)=acd101(93)+acd101(94)
      acd101(93)=acd101(68)*acd101(93)
      acd101(94)=acd101(51)*acd101(24)
      acd101(95)=acd101(57)*acd101(18)
      acd101(94)=acd101(94)+acd101(95)
      acd101(94)=acd101(69)*acd101(94)
      acd101(95)=acd101(53)*acd101(24)
      acd101(96)=acd101(58)*acd101(18)
      acd101(95)=acd101(95)+acd101(96)
      acd101(95)=acd101(70)*acd101(95)
      acd101(96)=acd101(55)*acd101(24)
      acd101(97)=acd101(59)*acd101(18)
      acd101(96)=acd101(96)+acd101(97)
      acd101(96)=acd101(71)*acd101(96)
      brack=2.0_ki*acd101(72)+acd101(73)+acd101(74)+acd101(75)+acd101(76)+acd10&
      &1(77)+acd101(78)+acd101(79)+acd101(80)+acd101(81)+acd101(82)+acd101(83)+&
      &acd101(84)+acd101(85)+acd101(86)+acd101(87)+acd101(88)+acd101(89)+acd101&
      &(90)+acd101(91)+acd101(92)+acd101(93)+acd101(94)+acd101(95)+acd101(96)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd101h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(45) :: acd101
      complex(ki) :: brack
      acd101(1)=d(iv1,iv2)
      acd101(2)=spvak2e2(iv3)
      acd101(3)=abb101(24)
      acd101(4)=spvae2l4(iv3)
      acd101(5)=abb101(18)
      acd101(6)=spvae2l5(iv3)
      acd101(7)=abb101(40)
      acd101(8)=spvae1e2(iv3)
      acd101(9)=abb101(36)
      acd101(10)=spvae2e1(iv3)
      acd101(11)=abb101(30)
      acd101(12)=d(iv1,iv3)
      acd101(13)=spvak2e2(iv2)
      acd101(14)=spvae2l4(iv2)
      acd101(15)=spvae2l5(iv2)
      acd101(16)=spvae1e2(iv2)
      acd101(17)=spvae2e1(iv2)
      acd101(18)=d(iv2,iv3)
      acd101(19)=spvak2e2(iv1)
      acd101(20)=spvae2l4(iv1)
      acd101(21)=spvae2l5(iv1)
      acd101(22)=spvae1e2(iv1)
      acd101(23)=spvae2e1(iv1)
      acd101(24)=spvae1l4(iv3)
      acd101(25)=abb101(46)
      acd101(26)=spvae1l5(iv3)
      acd101(27)=abb101(45)
      acd101(28)=spvae1l4(iv2)
      acd101(29)=spvae1l5(iv2)
      acd101(30)=spvae1l4(iv1)
      acd101(31)=spvae1l5(iv1)
      acd101(32)=spvak2e1(iv3)
      acd101(33)=spvak2e1(iv2)
      acd101(34)=spvak2e1(iv1)
      acd101(35)=-acd101(2)*acd101(1)
      acd101(36)=-acd101(13)*acd101(12)
      acd101(37)=-acd101(19)*acd101(18)
      acd101(35)=acd101(37)+acd101(35)+acd101(36)
      acd101(35)=acd101(3)*acd101(35)
      acd101(36)=-acd101(8)*acd101(1)
      acd101(37)=-acd101(16)*acd101(12)
      acd101(38)=-acd101(22)*acd101(18)
      acd101(36)=acd101(38)+acd101(36)+acd101(37)
      acd101(36)=acd101(9)*acd101(36)
      acd101(37)=-acd101(10)*acd101(1)
      acd101(38)=-acd101(17)*acd101(12)
      acd101(39)=-acd101(23)*acd101(18)
      acd101(37)=acd101(39)+acd101(37)+acd101(38)
      acd101(37)=acd101(11)*acd101(37)
      acd101(35)=acd101(37)+acd101(35)+acd101(36)
      acd101(36)=acd101(19)*acd101(17)
      acd101(37)=acd101(23)*acd101(13)
      acd101(36)=acd101(36)+acd101(37)
      acd101(37)=acd101(24)*acd101(36)
      acd101(38)=acd101(19)*acd101(10)
      acd101(39)=acd101(23)*acd101(2)
      acd101(38)=acd101(38)+acd101(39)
      acd101(39)=acd101(28)*acd101(38)
      acd101(40)=acd101(13)*acd101(10)
      acd101(41)=acd101(17)*acd101(2)
      acd101(40)=acd101(40)+acd101(41)
      acd101(41)=acd101(30)*acd101(40)
      acd101(37)=acd101(41)+acd101(39)+acd101(37)
      acd101(37)=acd101(25)*acd101(37)
      acd101(36)=acd101(26)*acd101(36)
      acd101(38)=acd101(29)*acd101(38)
      acd101(39)=acd101(31)*acd101(40)
      acd101(36)=acd101(39)+acd101(38)+acd101(36)
      acd101(36)=acd101(27)*acd101(36)
      acd101(38)=acd101(33)*acd101(22)
      acd101(39)=acd101(34)*acd101(16)
      acd101(38)=acd101(38)+acd101(39)
      acd101(39)=acd101(25)*acd101(38)
      acd101(40)=2.0_ki*acd101(5)
      acd101(41)=-acd101(1)*acd101(40)
      acd101(39)=acd101(41)+acd101(39)
      acd101(39)=acd101(4)*acd101(39)
      acd101(38)=acd101(27)*acd101(38)
      acd101(41)=2.0_ki*acd101(7)
      acd101(42)=-acd101(1)*acd101(41)
      acd101(38)=acd101(42)+acd101(38)
      acd101(38)=acd101(6)*acd101(38)
      acd101(42)=acd101(32)*acd101(22)
      acd101(43)=acd101(34)*acd101(8)
      acd101(42)=acd101(42)+acd101(43)
      acd101(43)=acd101(25)*acd101(42)
      acd101(44)=-acd101(12)*acd101(40)
      acd101(43)=acd101(44)+acd101(43)
      acd101(43)=acd101(14)*acd101(43)
      acd101(42)=acd101(27)*acd101(42)
      acd101(44)=-acd101(12)*acd101(41)
      acd101(42)=acd101(44)+acd101(42)
      acd101(42)=acd101(15)*acd101(42)
      acd101(44)=acd101(32)*acd101(16)
      acd101(45)=acd101(33)*acd101(8)
      acd101(44)=acd101(44)+acd101(45)
      acd101(45)=acd101(25)*acd101(44)
      acd101(40)=-acd101(18)*acd101(40)
      acd101(40)=acd101(40)+acd101(45)
      acd101(40)=acd101(20)*acd101(40)
      acd101(44)=acd101(27)*acd101(44)
      acd101(41)=-acd101(18)*acd101(41)
      acd101(41)=acd101(41)+acd101(44)
      acd101(41)=acd101(21)*acd101(41)
      brack=2.0_ki*acd101(35)+acd101(36)+acd101(37)+acd101(38)+acd101(39)+acd10&
      &1(40)+acd101(41)+acd101(42)+acd101(43)
   end function brack_4
!---#] function brack_4:
!---#[ function brack_5:
   pure function brack_5(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd101h12
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd101
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_5
!---#] function brack_5:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3,i4) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd101h12
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
      qshift = k2-k4
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
end module     p2_gg_httbar_d101h12l1d
