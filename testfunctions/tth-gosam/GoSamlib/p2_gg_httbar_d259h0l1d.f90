module     p2_gg_httbar_d259h0l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d259h0l1d.f90
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
      use p2_gg_httbar_abbrevd259h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(71) :: acd259
      complex(ki) :: brack
      acd259(1)=dotproduct(qshift,qshift)
      acd259(2)=abb259(54)
      acd259(3)=dotproduct(qshift,spval4k1)
      acd259(4)=abb259(40)
      acd259(5)=dotproduct(qshift,spval5k1)
      acd259(6)=abb259(14)
      acd259(7)=dotproduct(qshift,spvak2e1)
      acd259(8)=abb259(32)
      acd259(9)=dotproduct(qshift,spvae1k2)
      acd259(10)=abb259(18)
      acd259(11)=dotproduct(qshift,spvae2k2)
      acd259(12)=abb259(53)
      acd259(13)=dotproduct(qshift,spval3e1)
      acd259(14)=abb259(57)
      acd259(15)=dotproduct(qshift,spvae1l3)
      acd259(16)=abb259(37)
      acd259(17)=dotproduct(qshift,spval4e2)
      acd259(18)=abb259(15)
      acd259(19)=dotproduct(qshift,spval5e1)
      acd259(20)=abb259(33)
      acd259(21)=abb259(31)
      acd259(22)=abb259(23)
      acd259(23)=abb259(42)
      acd259(24)=abb259(13)
      acd259(25)=abb259(12)
      acd259(26)=abb259(39)
      acd259(27)=abb259(9)
      acd259(28)=abb259(7)
      acd259(29)=abb259(34)
      acd259(30)=abb259(27)
      acd259(31)=abb259(25)
      acd259(32)=abb259(17)
      acd259(33)=dotproduct(qshift,spvae2e1)
      acd259(34)=abb259(41)
      acd259(35)=abb259(20)
      acd259(36)=abb259(19)
      acd259(37)=abb259(16)
      acd259(38)=abb259(63)
      acd259(39)=dotproduct(qshift,spvae1e2)
      acd259(40)=abb259(62)
      acd259(41)=abb259(48)
      acd259(42)=abb259(61)
      acd259(43)=abb259(8)
      acd259(44)=abb259(26)
      acd259(45)=abb259(21)
      acd259(46)=abb259(58)
      acd259(47)=abb259(35)
      acd259(48)=abb259(29)
      acd259(49)=abb259(45)
      acd259(50)=abb259(10)
      acd259(51)=abb259(36)
      acd259(52)=abb259(30)
      acd259(53)=abb259(46)
      acd259(54)=dotproduct(qshift,spval4e1)
      acd259(55)=abb259(24)
      acd259(56)=abb259(22)
      acd259(57)=abb259(38)
      acd259(58)=abb259(11)
      acd259(59)=-acd259(5)*acd259(6)
      acd259(60)=-acd259(3)*acd259(4)
      acd259(61)=-acd259(19)*acd259(20)
      acd259(62)=-acd259(17)*acd259(18)
      acd259(63)=-acd259(15)*acd259(16)
      acd259(64)=-acd259(13)*acd259(14)
      acd259(65)=-acd259(11)*acd259(12)
      acd259(66)=-acd259(9)*acd259(10)
      acd259(67)=-acd259(7)*acd259(8)
      acd259(68)=acd259(1)*acd259(2)
      acd259(59)=acd259(68)+acd259(67)+acd259(66)+acd259(65)+acd259(64)+acd259(&
      &63)+acd259(62)+acd259(61)+acd259(60)+acd259(21)+acd259(59)
      acd259(59)=acd259(1)*acd259(59)
      acd259(60)=acd259(5)*acd259(25)
      acd259(61)=acd259(3)*acd259(22)
      acd259(62)=acd259(15)*acd259(30)
      acd259(63)=acd259(11)*acd259(29)
      acd259(64)=acd259(9)*acd259(28)
      acd259(60)=acd259(64)+acd259(63)+acd259(62)+acd259(61)-acd259(31)+acd259(&
      &60)
      acd259(60)=acd259(7)*acd259(60)
      acd259(61)=acd259(5)*acd259(26)
      acd259(62)=acd259(3)*acd259(23)
      acd259(63)=acd259(15)*acd259(44)
      acd259(61)=acd259(63)+acd259(62)-acd259(45)+acd259(61)
      acd259(61)=acd259(13)*acd259(61)
      acd259(62)=acd259(39)*acd259(42)
      acd259(63)=-acd259(39)*acd259(40)
      acd259(63)=acd259(41)+acd259(63)
      acd259(63)=acd259(19)*acd259(63)
      acd259(64)=acd259(13)*acd259(38)
      acd259(62)=acd259(64)+acd259(63)-acd259(43)+acd259(62)
      acd259(62)=acd259(11)*acd259(62)
      acd259(63)=acd259(33)*acd259(36)
      acd259(64)=-acd259(33)*acd259(34)
      acd259(64)=acd259(35)+acd259(64)
      acd259(64)=acd259(17)*acd259(64)
      acd259(65)=acd259(13)*acd259(32)
      acd259(63)=acd259(65)+acd259(64)-acd259(37)+acd259(63)
      acd259(63)=acd259(9)*acd259(63)
      acd259(64)=acd259(19)*acd259(47)
      acd259(65)=acd259(17)*acd259(46)
      acd259(64)=acd259(65)-acd259(48)+acd259(64)
      acd259(64)=acd259(15)*acd259(64)
      acd259(65)=-acd259(54)*acd259(57)
      acd259(66)=-acd259(33)*acd259(53)
      acd259(67)=-acd259(5)*acd259(27)
      acd259(68)=-acd259(3)*acd259(24)
      acd259(69)=acd259(54)*acd259(55)
      acd259(69)=-acd259(56)+acd259(69)
      acd259(69)=acd259(39)*acd259(69)
      acd259(70)=acd259(39)*acd259(51)
      acd259(70)=-acd259(52)+acd259(70)
      acd259(70)=acd259(19)*acd259(70)
      acd259(71)=acd259(33)*acd259(49)
      acd259(71)=-acd259(50)+acd259(71)
      acd259(71)=acd259(17)*acd259(71)
      brack=acd259(58)+acd259(59)+acd259(60)+acd259(61)+acd259(62)+acd259(63)+a&
      &cd259(64)+acd259(65)+acd259(66)+acd259(67)+acd259(68)+acd259(69)+acd259(&
      &70)+acd259(71)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd259h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(90) :: acd259
      complex(ki) :: brack
      acd259(1)=qshift(iv1)
      acd259(2)=dotproduct(qshift,qshift)
      acd259(3)=abb259(54)
      acd259(4)=dotproduct(qshift,spval4k1)
      acd259(5)=abb259(40)
      acd259(6)=dotproduct(qshift,spval5k1)
      acd259(7)=abb259(14)
      acd259(8)=dotproduct(qshift,spvak2e1)
      acd259(9)=abb259(32)
      acd259(10)=dotproduct(qshift,spvae1k2)
      acd259(11)=abb259(18)
      acd259(12)=dotproduct(qshift,spvae2k2)
      acd259(13)=abb259(53)
      acd259(14)=dotproduct(qshift,spval3e1)
      acd259(15)=abb259(57)
      acd259(16)=dotproduct(qshift,spvae1l3)
      acd259(17)=abb259(37)
      acd259(18)=dotproduct(qshift,spval4e2)
      acd259(19)=abb259(15)
      acd259(20)=dotproduct(qshift,spval5e1)
      acd259(21)=abb259(33)
      acd259(22)=abb259(31)
      acd259(23)=spval4k1(iv1)
      acd259(24)=abb259(23)
      acd259(25)=abb259(42)
      acd259(26)=abb259(13)
      acd259(27)=spval5k1(iv1)
      acd259(28)=abb259(12)
      acd259(29)=abb259(39)
      acd259(30)=abb259(9)
      acd259(31)=spvak2e1(iv1)
      acd259(32)=abb259(7)
      acd259(33)=abb259(34)
      acd259(34)=abb259(27)
      acd259(35)=abb259(25)
      acd259(36)=spvae1k2(iv1)
      acd259(37)=abb259(17)
      acd259(38)=dotproduct(qshift,spvae2e1)
      acd259(39)=abb259(41)
      acd259(40)=abb259(20)
      acd259(41)=abb259(19)
      acd259(42)=abb259(16)
      acd259(43)=spvae2k2(iv1)
      acd259(44)=abb259(63)
      acd259(45)=dotproduct(qshift,spvae1e2)
      acd259(46)=abb259(62)
      acd259(47)=abb259(48)
      acd259(48)=abb259(61)
      acd259(49)=abb259(8)
      acd259(50)=spval3e1(iv1)
      acd259(51)=abb259(26)
      acd259(52)=abb259(21)
      acd259(53)=spvae1l3(iv1)
      acd259(54)=abb259(58)
      acd259(55)=abb259(35)
      acd259(56)=abb259(29)
      acd259(57)=spval4e2(iv1)
      acd259(58)=abb259(45)
      acd259(59)=abb259(10)
      acd259(60)=spval5e1(iv1)
      acd259(61)=abb259(36)
      acd259(62)=abb259(30)
      acd259(63)=spvae2e1(iv1)
      acd259(64)=abb259(46)
      acd259(65)=spvae1e2(iv1)
      acd259(66)=dotproduct(qshift,spval4e1)
      acd259(67)=abb259(24)
      acd259(68)=abb259(22)
      acd259(69)=spval4e1(iv1)
      acd259(70)=abb259(38)
      acd259(71)=acd259(6)*acd259(7)
      acd259(72)=acd259(4)*acd259(5)
      acd259(73)=acd259(16)*acd259(17)
      acd259(74)=acd259(20)*acd259(21)
      acd259(75)=acd259(18)*acd259(19)
      acd259(76)=acd259(14)*acd259(15)
      acd259(77)=acd259(8)*acd259(9)
      acd259(78)=acd259(12)*acd259(13)
      acd259(79)=acd259(10)*acd259(11)
      acd259(80)=acd259(2)*acd259(3)
      acd259(71)=-2.0_ki*acd259(80)+acd259(79)+acd259(78)+acd259(77)+acd259(76)&
      &+acd259(75)+acd259(74)+acd259(73)+acd259(72)-acd259(22)+acd259(71)
      acd259(71)=acd259(1)*acd259(71)
      acd259(72)=acd259(27)*acd259(7)
      acd259(73)=acd259(23)*acd259(5)
      acd259(74)=acd259(60)*acd259(21)
      acd259(75)=acd259(57)*acd259(19)
      acd259(76)=acd259(53)*acd259(17)
      acd259(77)=acd259(50)*acd259(15)
      acd259(78)=acd259(43)*acd259(13)
      acd259(79)=acd259(36)*acd259(11)
      acd259(80)=acd259(31)*acd259(9)
      acd259(72)=acd259(80)+acd259(79)+acd259(78)+acd259(77)+acd259(76)+acd259(&
      &75)+acd259(74)+acd259(72)+acd259(73)
      acd259(72)=acd259(2)*acd259(72)
      acd259(73)=-acd259(45)*acd259(48)
      acd259(74)=acd259(45)*acd259(46)
      acd259(74)=acd259(74)-acd259(47)
      acd259(75)=acd259(20)*acd259(74)
      acd259(76)=-acd259(14)*acd259(44)
      acd259(77)=-acd259(8)*acd259(33)
      acd259(73)=acd259(77)+acd259(76)+acd259(75)+acd259(49)+acd259(73)
      acd259(73)=acd259(43)*acd259(73)
      acd259(75)=-acd259(38)*acd259(41)
      acd259(76)=acd259(38)*acd259(39)
      acd259(76)=acd259(76)-acd259(40)
      acd259(77)=acd259(18)*acd259(76)
      acd259(78)=-acd259(14)*acd259(37)
      acd259(79)=-acd259(8)*acd259(32)
      acd259(75)=acd259(79)+acd259(78)+acd259(77)+acd259(42)+acd259(75)
      acd259(75)=acd259(36)*acd259(75)
      acd259(77)=acd259(20)*acd259(46)
      acd259(77)=acd259(77)-acd259(48)
      acd259(77)=acd259(65)*acd259(77)
      acd259(74)=acd259(60)*acd259(74)
      acd259(78)=-acd259(50)*acd259(44)
      acd259(79)=-acd259(31)*acd259(33)
      acd259(74)=acd259(79)+acd259(78)+acd259(74)+acd259(77)
      acd259(74)=acd259(12)*acd259(74)
      acd259(77)=acd259(18)*acd259(39)
      acd259(77)=acd259(77)-acd259(41)
      acd259(77)=acd259(63)*acd259(77)
      acd259(76)=acd259(57)*acd259(76)
      acd259(78)=-acd259(50)*acd259(37)
      acd259(79)=-acd259(31)*acd259(32)
      acd259(76)=acd259(79)+acd259(78)+acd259(76)+acd259(77)
      acd259(76)=acd259(10)*acd259(76)
      acd259(77)=-acd259(27)*acd259(29)
      acd259(78)=-acd259(23)*acd259(25)
      acd259(79)=-acd259(53)*acd259(51)
      acd259(77)=acd259(79)+acd259(77)+acd259(78)
      acd259(77)=acd259(14)*acd259(77)
      acd259(78)=-acd259(27)*acd259(28)
      acd259(79)=-acd259(23)*acd259(24)
      acd259(80)=-acd259(53)*acd259(34)
      acd259(78)=acd259(80)+acd259(78)+acd259(79)
      acd259(78)=acd259(8)*acd259(78)
      acd259(79)=-acd259(6)*acd259(29)
      acd259(80)=-acd259(4)*acd259(25)
      acd259(81)=-acd259(16)*acd259(51)
      acd259(79)=acd259(81)+acd259(80)+acd259(52)+acd259(79)
      acd259(79)=acd259(50)*acd259(79)
      acd259(80)=-acd259(6)*acd259(28)
      acd259(81)=-acd259(4)*acd259(24)
      acd259(82)=-acd259(16)*acd259(34)
      acd259(80)=acd259(82)+acd259(81)+acd259(35)+acd259(80)
      acd259(80)=acd259(31)*acd259(80)
      acd259(81)=-acd259(45)*acd259(61)
      acd259(82)=-acd259(16)*acd259(55)
      acd259(81)=acd259(82)+acd259(62)+acd259(81)
      acd259(81)=acd259(60)*acd259(81)
      acd259(82)=-acd259(38)*acd259(58)
      acd259(83)=-acd259(16)*acd259(54)
      acd259(82)=acd259(83)+acd259(59)+acd259(82)
      acd259(82)=acd259(57)*acd259(82)
      acd259(83)=-acd259(65)*acd259(61)
      acd259(84)=-acd259(53)*acd259(55)
      acd259(83)=acd259(83)+acd259(84)
      acd259(83)=acd259(20)*acd259(83)
      acd259(84)=-acd259(63)*acd259(58)
      acd259(85)=-acd259(53)*acd259(54)
      acd259(84)=acd259(84)+acd259(85)
      acd259(84)=acd259(18)*acd259(84)
      acd259(85)=-acd259(45)*acd259(67)
      acd259(85)=acd259(85)+acd259(70)
      acd259(85)=acd259(69)*acd259(85)
      acd259(86)=acd259(63)*acd259(64)
      acd259(87)=acd259(27)*acd259(30)
      acd259(88)=acd259(23)*acd259(26)
      acd259(89)=-acd259(67)*acd259(66)
      acd259(89)=acd259(68)+acd259(89)
      acd259(89)=acd259(65)*acd259(89)
      acd259(90)=acd259(53)*acd259(56)
      brack=2.0_ki*acd259(71)+acd259(72)+acd259(73)+acd259(74)+acd259(75)+acd25&
      &9(76)+acd259(77)+acd259(78)+acd259(79)+acd259(80)+acd259(81)+acd259(82)+&
      &acd259(83)+acd259(84)+acd259(85)+acd259(86)+acd259(87)+acd259(88)+acd259&
      &(89)+acd259(90)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd259h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(86) :: acd259
      complex(ki) :: brack
      acd259(1)=d(iv1,iv2)
      acd259(2)=dotproduct(qshift,qshift)
      acd259(3)=abb259(54)
      acd259(4)=dotproduct(qshift,spval4k1)
      acd259(5)=abb259(40)
      acd259(6)=dotproduct(qshift,spval5k1)
      acd259(7)=abb259(14)
      acd259(8)=dotproduct(qshift,spvak2e1)
      acd259(9)=abb259(32)
      acd259(10)=dotproduct(qshift,spvae1k2)
      acd259(11)=abb259(18)
      acd259(12)=dotproduct(qshift,spvae2k2)
      acd259(13)=abb259(53)
      acd259(14)=dotproduct(qshift,spval3e1)
      acd259(15)=abb259(57)
      acd259(16)=dotproduct(qshift,spvae1l3)
      acd259(17)=abb259(37)
      acd259(18)=dotproduct(qshift,spval4e2)
      acd259(19)=abb259(15)
      acd259(20)=dotproduct(qshift,spval5e1)
      acd259(21)=abb259(33)
      acd259(22)=abb259(31)
      acd259(23)=qshift(iv1)
      acd259(24)=qshift(iv2)
      acd259(25)=spval4k1(iv2)
      acd259(26)=spval5k1(iv2)
      acd259(27)=spvak2e1(iv2)
      acd259(28)=spvae1k2(iv2)
      acd259(29)=spvae2k2(iv2)
      acd259(30)=spval3e1(iv2)
      acd259(31)=spvae1l3(iv2)
      acd259(32)=spval4e2(iv2)
      acd259(33)=spval5e1(iv2)
      acd259(34)=spval4k1(iv1)
      acd259(35)=spval5k1(iv1)
      acd259(36)=spvak2e1(iv1)
      acd259(37)=spvae1k2(iv1)
      acd259(38)=spvae2k2(iv1)
      acd259(39)=spval3e1(iv1)
      acd259(40)=spvae1l3(iv1)
      acd259(41)=spval4e2(iv1)
      acd259(42)=spval5e1(iv1)
      acd259(43)=abb259(23)
      acd259(44)=abb259(42)
      acd259(45)=abb259(12)
      acd259(46)=abb259(39)
      acd259(47)=abb259(7)
      acd259(48)=abb259(34)
      acd259(49)=abb259(27)
      acd259(50)=abb259(17)
      acd259(51)=dotproduct(qshift,spvae2e1)
      acd259(52)=abb259(41)
      acd259(53)=abb259(20)
      acd259(54)=spvae2e1(iv2)
      acd259(55)=abb259(19)
      acd259(56)=spvae2e1(iv1)
      acd259(57)=abb259(63)
      acd259(58)=dotproduct(qshift,spvae1e2)
      acd259(59)=abb259(62)
      acd259(60)=abb259(48)
      acd259(61)=spvae1e2(iv2)
      acd259(62)=abb259(61)
      acd259(63)=spvae1e2(iv1)
      acd259(64)=abb259(26)
      acd259(65)=abb259(58)
      acd259(66)=abb259(35)
      acd259(67)=abb259(45)
      acd259(68)=abb259(36)
      acd259(69)=spval4e1(iv2)
      acd259(70)=abb259(24)
      acd259(71)=spval4e1(iv1)
      acd259(72)=-acd259(7)*acd259(26)
      acd259(73)=-acd259(5)*acd259(25)
      acd259(74)=-acd259(31)*acd259(17)
      acd259(75)=-acd259(33)*acd259(21)
      acd259(76)=-acd259(32)*acd259(19)
      acd259(77)=-acd259(30)*acd259(15)
      acd259(78)=-acd259(27)*acd259(9)
      acd259(79)=-acd259(29)*acd259(13)
      acd259(80)=-acd259(28)*acd259(11)
      acd259(81)=acd259(24)*acd259(3)
      acd259(72)=4.0_ki*acd259(81)+acd259(80)+acd259(79)+acd259(78)+acd259(77)+&
      &acd259(76)+acd259(75)+acd259(74)+acd259(72)+acd259(73)
      acd259(72)=acd259(23)*acd259(72)
      acd259(73)=acd259(3)*acd259(2)
      acd259(74)=-acd259(20)*acd259(21)
      acd259(75)=-acd259(18)*acd259(19)
      acd259(76)=-acd259(17)*acd259(16)
      acd259(77)=-acd259(15)*acd259(14)
      acd259(78)=-acd259(12)*acd259(13)
      acd259(79)=-acd259(10)*acd259(11)
      acd259(80)=-acd259(9)*acd259(8)
      acd259(81)=-acd259(7)*acd259(6)
      acd259(82)=-acd259(5)*acd259(4)
      acd259(73)=acd259(82)+acd259(81)+acd259(80)+acd259(79)+acd259(78)+acd259(&
      &77)+acd259(76)+acd259(75)+acd259(74)+acd259(22)+2.0_ki*acd259(73)
      acd259(73)=acd259(1)*acd259(73)
      acd259(74)=-acd259(7)*acd259(35)
      acd259(75)=-acd259(5)*acd259(34)
      acd259(76)=-acd259(40)*acd259(17)
      acd259(77)=-acd259(42)*acd259(21)
      acd259(78)=-acd259(41)*acd259(19)
      acd259(79)=-acd259(39)*acd259(15)
      acd259(80)=-acd259(36)*acd259(9)
      acd259(81)=-acd259(38)*acd259(13)
      acd259(82)=-acd259(37)*acd259(11)
      acd259(74)=acd259(82)+acd259(81)+acd259(80)+acd259(79)+acd259(78)+acd259(&
      &77)+acd259(76)+acd259(74)+acd259(75)
      acd259(74)=acd259(24)*acd259(74)
      acd259(72)=acd259(73)+acd259(74)+acd259(72)
      acd259(73)=acd259(61)*acd259(62)
      acd259(74)=acd259(59)*acd259(61)
      acd259(75)=-acd259(20)*acd259(74)
      acd259(76)=acd259(59)*acd259(58)
      acd259(76)=acd259(76)-acd259(60)
      acd259(77)=-acd259(33)*acd259(76)
      acd259(78)=acd259(30)*acd259(57)
      acd259(79)=acd259(27)*acd259(48)
      acd259(73)=acd259(79)+acd259(78)+acd259(77)+acd259(73)+acd259(75)
      acd259(73)=acd259(38)*acd259(73)
      acd259(75)=acd259(54)*acd259(55)
      acd259(77)=acd259(52)*acd259(54)
      acd259(78)=-acd259(18)*acd259(77)
      acd259(79)=acd259(52)*acd259(51)
      acd259(79)=acd259(79)-acd259(53)
      acd259(80)=-acd259(32)*acd259(79)
      acd259(81)=acd259(30)*acd259(50)
      acd259(82)=acd259(27)*acd259(47)
      acd259(75)=acd259(82)+acd259(81)+acd259(80)+acd259(75)+acd259(78)
      acd259(75)=acd259(37)*acd259(75)
      acd259(78)=acd259(63)*acd259(62)
      acd259(80)=acd259(59)*acd259(63)
      acd259(81)=-acd259(20)*acd259(80)
      acd259(76)=-acd259(42)*acd259(76)
      acd259(82)=acd259(39)*acd259(57)
      acd259(83)=acd259(36)*acd259(48)
      acd259(76)=acd259(83)+acd259(82)+acd259(76)+acd259(78)+acd259(81)
      acd259(76)=acd259(29)*acd259(76)
      acd259(78)=acd259(56)*acd259(55)
      acd259(81)=acd259(52)*acd259(56)
      acd259(82)=-acd259(18)*acd259(81)
      acd259(79)=-acd259(41)*acd259(79)
      acd259(83)=acd259(39)*acd259(50)
      acd259(84)=acd259(36)*acd259(47)
      acd259(78)=acd259(84)+acd259(83)+acd259(79)+acd259(78)+acd259(82)
      acd259(78)=acd259(28)*acd259(78)
      acd259(79)=acd259(61)*acd259(68)
      acd259(82)=acd259(31)*acd259(66)
      acd259(74)=-acd259(12)*acd259(74)
      acd259(74)=acd259(74)+acd259(79)+acd259(82)
      acd259(74)=acd259(42)*acd259(74)
      acd259(79)=acd259(54)*acd259(67)
      acd259(82)=acd259(31)*acd259(65)
      acd259(77)=-acd259(10)*acd259(77)
      acd259(77)=acd259(77)+acd259(79)+acd259(82)
      acd259(77)=acd259(41)*acd259(77)
      acd259(79)=acd259(26)*acd259(46)
      acd259(82)=acd259(25)*acd259(44)
      acd259(83)=acd259(31)*acd259(64)
      acd259(79)=acd259(83)+acd259(79)+acd259(82)
      acd259(79)=acd259(39)*acd259(79)
      acd259(82)=acd259(26)*acd259(45)
      acd259(83)=acd259(25)*acd259(43)
      acd259(84)=acd259(31)*acd259(49)
      acd259(82)=acd259(84)+acd259(82)+acd259(83)
      acd259(82)=acd259(36)*acd259(82)
      acd259(83)=acd259(63)*acd259(68)
      acd259(84)=acd259(40)*acd259(66)
      acd259(80)=-acd259(12)*acd259(80)
      acd259(80)=acd259(80)+acd259(83)+acd259(84)
      acd259(80)=acd259(33)*acd259(80)
      acd259(83)=acd259(56)*acd259(67)
      acd259(84)=acd259(40)*acd259(65)
      acd259(81)=-acd259(10)*acd259(81)
      acd259(81)=acd259(81)+acd259(83)+acd259(84)
      acd259(81)=acd259(32)*acd259(81)
      acd259(83)=acd259(35)*acd259(46)
      acd259(84)=acd259(34)*acd259(44)
      acd259(85)=acd259(40)*acd259(64)
      acd259(83)=acd259(85)+acd259(83)+acd259(84)
      acd259(83)=acd259(30)*acd259(83)
      acd259(84)=acd259(35)*acd259(45)
      acd259(85)=acd259(34)*acd259(43)
      acd259(86)=acd259(40)*acd259(49)
      acd259(84)=acd259(86)+acd259(84)+acd259(85)
      acd259(84)=acd259(27)*acd259(84)
      acd259(85)=acd259(63)*acd259(69)
      acd259(86)=acd259(61)*acd259(71)
      acd259(85)=acd259(85)+acd259(86)
      acd259(85)=acd259(70)*acd259(85)
      brack=2.0_ki*acd259(72)+acd259(73)+acd259(74)+acd259(75)+acd259(76)+acd25&
      &9(77)+acd259(78)+acd259(79)+acd259(80)+acd259(81)+acd259(82)+acd259(83)+&
      &acd259(84)+acd259(85)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd259h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(63) :: acd259
      complex(ki) :: brack
      acd259(1)=d(iv1,iv2)
      acd259(2)=qshift(iv3)
      acd259(3)=abb259(54)
      acd259(4)=spval4k1(iv3)
      acd259(5)=abb259(40)
      acd259(6)=spval5k1(iv3)
      acd259(7)=abb259(14)
      acd259(8)=spvak2e1(iv3)
      acd259(9)=abb259(32)
      acd259(10)=spvae1k2(iv3)
      acd259(11)=abb259(18)
      acd259(12)=spvae2k2(iv3)
      acd259(13)=abb259(53)
      acd259(14)=spval3e1(iv3)
      acd259(15)=abb259(57)
      acd259(16)=spvae1l3(iv3)
      acd259(17)=abb259(37)
      acd259(18)=spval4e2(iv3)
      acd259(19)=abb259(15)
      acd259(20)=spval5e1(iv3)
      acd259(21)=abb259(33)
      acd259(22)=d(iv1,iv3)
      acd259(23)=qshift(iv2)
      acd259(24)=spval4k1(iv2)
      acd259(25)=spval5k1(iv2)
      acd259(26)=spvak2e1(iv2)
      acd259(27)=spvae1k2(iv2)
      acd259(28)=spvae2k2(iv2)
      acd259(29)=spval3e1(iv2)
      acd259(30)=spvae1l3(iv2)
      acd259(31)=spval4e2(iv2)
      acd259(32)=spval5e1(iv2)
      acd259(33)=d(iv2,iv3)
      acd259(34)=qshift(iv1)
      acd259(35)=spval4k1(iv1)
      acd259(36)=spval5k1(iv1)
      acd259(37)=spvak2e1(iv1)
      acd259(38)=spvae1k2(iv1)
      acd259(39)=spvae2k2(iv1)
      acd259(40)=spval3e1(iv1)
      acd259(41)=spvae1l3(iv1)
      acd259(42)=spval4e2(iv1)
      acd259(43)=spval5e1(iv1)
      acd259(44)=spvae2e1(iv3)
      acd259(45)=abb259(41)
      acd259(46)=spvae2e1(iv2)
      acd259(47)=spvae2e1(iv1)
      acd259(48)=spvae1e2(iv3)
      acd259(49)=abb259(62)
      acd259(50)=spvae1e2(iv2)
      acd259(51)=spvae1e2(iv1)
      acd259(52)=acd259(21)*acd259(43)
      acd259(53)=acd259(19)*acd259(42)
      acd259(54)=acd259(17)*acd259(41)
      acd259(55)=acd259(15)*acd259(40)
      acd259(56)=acd259(13)*acd259(39)
      acd259(57)=acd259(11)*acd259(38)
      acd259(58)=acd259(9)*acd259(37)
      acd259(59)=acd259(7)*acd259(36)
      acd259(60)=acd259(5)*acd259(35)
      acd259(61)=4.0_ki*acd259(3)
      acd259(62)=-acd259(34)*acd259(61)
      acd259(52)=acd259(62)+acd259(60)+acd259(59)+acd259(58)+acd259(57)+acd259(&
      &56)+acd259(55)+acd259(54)+acd259(52)+acd259(53)
      acd259(52)=acd259(33)*acd259(52)
      acd259(53)=acd259(21)*acd259(32)
      acd259(54)=acd259(19)*acd259(31)
      acd259(55)=acd259(17)*acd259(30)
      acd259(56)=acd259(15)*acd259(29)
      acd259(57)=acd259(13)*acd259(28)
      acd259(58)=acd259(11)*acd259(27)
      acd259(59)=acd259(9)*acd259(26)
      acd259(60)=acd259(7)*acd259(25)
      acd259(62)=acd259(5)*acd259(24)
      acd259(63)=-acd259(23)*acd259(61)
      acd259(53)=acd259(63)+acd259(62)+acd259(60)+acd259(59)+acd259(58)+acd259(&
      &57)+acd259(56)+acd259(55)+acd259(53)+acd259(54)
      acd259(53)=acd259(22)*acd259(53)
      acd259(54)=acd259(20)*acd259(21)
      acd259(55)=acd259(18)*acd259(19)
      acd259(56)=acd259(17)*acd259(16)
      acd259(57)=acd259(15)*acd259(14)
      acd259(58)=acd259(12)*acd259(13)
      acd259(59)=acd259(10)*acd259(11)
      acd259(60)=acd259(9)*acd259(8)
      acd259(62)=acd259(7)*acd259(6)
      acd259(63)=acd259(5)*acd259(4)
      acd259(61)=-acd259(2)*acd259(61)
      acd259(54)=acd259(61)+acd259(63)+acd259(62)+acd259(60)+acd259(59)+acd259(&
      &58)+acd259(57)+acd259(56)+acd259(54)+acd259(55)
      acd259(54)=acd259(1)*acd259(54)
      acd259(52)=acd259(54)+acd259(52)+acd259(53)
      acd259(53)=acd259(32)*acd259(39)
      acd259(54)=acd259(28)*acd259(43)
      acd259(53)=acd259(53)+acd259(54)
      acd259(53)=acd259(48)*acd259(53)
      acd259(54)=acd259(39)*acd259(50)
      acd259(55)=acd259(28)*acd259(51)
      acd259(54)=acd259(54)+acd259(55)
      acd259(54)=acd259(20)*acd259(54)
      acd259(55)=acd259(43)*acd259(50)
      acd259(56)=acd259(32)*acd259(51)
      acd259(55)=acd259(55)+acd259(56)
      acd259(55)=acd259(12)*acd259(55)
      acd259(53)=acd259(55)+acd259(54)+acd259(53)
      acd259(53)=acd259(49)*acd259(53)
      acd259(54)=acd259(31)*acd259(38)
      acd259(55)=acd259(27)*acd259(42)
      acd259(54)=acd259(54)+acd259(55)
      acd259(54)=acd259(44)*acd259(54)
      acd259(55)=acd259(38)*acd259(46)
      acd259(56)=acd259(27)*acd259(47)
      acd259(55)=acd259(55)+acd259(56)
      acd259(55)=acd259(18)*acd259(55)
      acd259(56)=acd259(42)*acd259(46)
      acd259(57)=acd259(31)*acd259(47)
      acd259(56)=acd259(56)+acd259(57)
      acd259(56)=acd259(10)*acd259(56)
      acd259(54)=acd259(56)+acd259(55)+acd259(54)
      acd259(54)=acd259(45)*acd259(54)
      brack=2.0_ki*acd259(52)+acd259(53)+acd259(54)
   end function brack_4
!---#] function brack_4:
!---#[ function brack_5:
   pure function brack_5(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd259h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(10) :: acd259
      complex(ki) :: brack
      acd259(1)=d(iv1,iv2)
      acd259(2)=d(iv3,iv4)
      acd259(3)=abb259(54)
      acd259(4)=d(iv1,iv3)
      acd259(5)=d(iv2,iv4)
      acd259(6)=d(iv1,iv4)
      acd259(7)=d(iv2,iv3)
      acd259(8)=acd259(2)*acd259(1)
      acd259(9)=acd259(5)*acd259(4)
      acd259(10)=acd259(7)*acd259(6)
      acd259(8)=acd259(10)+acd259(8)+acd259(9)
      brack=8.0_ki*acd259(8)*acd259(3)
   end function brack_5
!---#] function brack_5:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3,i4) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd259h0
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
      qshift = k2-k3-k5
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
end module     p2_gg_httbar_d259h0l1d
