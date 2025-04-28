module     p2_gg_httbar_d75h4l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d75h4l1d.f90
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
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd75h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(60) :: acd75
      complex(ki) :: brack
      acd75(1)=dotproduct(k2,qshift)
      acd75(2)=dotproduct(qshift,spvae1k2)
      acd75(3)=abb75(10)
      acd75(4)=dotproduct(qshift,qshift)
      acd75(5)=abb75(30)
      acd75(6)=dotproduct(qshift,spval4e1)
      acd75(7)=abb75(15)
      acd75(8)=dotproduct(qshift,spvae1l4)
      acd75(9)=abb75(39)
      acd75(10)=dotproduct(qshift,spval5e1)
      acd75(11)=abb75(11)
      acd75(12)=dotproduct(qshift,spvae1e2)
      acd75(13)=abb75(27)
      acd75(14)=dotproduct(qshift,spvae2e1)
      acd75(15)=abb75(28)
      acd75(16)=abb75(21)
      acd75(17)=abb75(29)
      acd75(18)=abb75(19)
      acd75(19)=abb75(22)
      acd75(20)=dotproduct(qshift,spvak2l4)
      acd75(21)=abb75(35)
      acd75(22)=dotproduct(qshift,spvak2e2)
      acd75(23)=abb75(23)
      acd75(24)=abb75(9)
      acd75(25)=abb75(18)
      acd75(26)=abb75(32)
      acd75(27)=abb75(31)
      acd75(28)=abb75(14)
      acd75(29)=abb75(13)
      acd75(30)=abb75(24)
      acd75(31)=abb75(20)
      acd75(32)=dotproduct(qshift,spval3k2)
      acd75(33)=dotproduct(qshift,spvae1l3)
      acd75(34)=dotproduct(qshift,spval3l4)
      acd75(35)=dotproduct(qshift,spval3e2)
      acd75(36)=abb75(25)
      acd75(37)=dotproduct(qshift,spval4k2)
      acd75(38)=dotproduct(qshift,spvak2e1)
      acd75(39)=abb75(34)
      acd75(40)=dotproduct(qshift,spval5k2)
      acd75(41)=abb75(33)
      acd75(42)=dotproduct(qshift,spvae2k2)
      acd75(43)=abb75(26)
      acd75(44)=abb75(17)
      acd75(45)=dotproduct(qshift,spval4l3)
      acd75(46)=dotproduct(qshift,spval3e1)
      acd75(47)=dotproduct(qshift,spval5l3)
      acd75(48)=dotproduct(qshift,spvae2l3)
      acd75(49)=abb75(16)
      acd75(50)=abb75(12)
      acd75(51)=acd75(5)*acd75(4)
      acd75(52)=acd75(3)*acd75(1)
      acd75(53)=-acd75(17)*acd75(6)
      acd75(54)=acd75(18)*acd75(10)
      acd75(55)=-acd75(19)*acd75(14)
      acd75(56)=acd75(21)*acd75(20)
      acd75(57)=acd75(23)*acd75(22)
      acd75(51)=-acd75(24)+acd75(57)+acd75(56)+acd75(55)+acd75(54)+acd75(53)+ac&
      &d75(52)+acd75(51)
      acd75(51)=acd75(2)*acd75(51)
      acd75(52)=acd75(7)*acd75(6)
      acd75(53)=acd75(9)*acd75(8)
      acd75(54)=-acd75(11)*acd75(10)
      acd75(55)=-acd75(13)*acd75(12)
      acd75(56)=acd75(15)*acd75(14)
      acd75(52)=acd75(16)+acd75(56)+acd75(55)+acd75(54)+acd75(53)+acd75(52)
      acd75(52)=acd75(4)*acd75(52)
      acd75(53)=acd75(32)*acd75(5)
      acd75(54)=acd75(34)*acd75(9)
      acd75(55)=-acd75(35)*acd75(13)
      acd75(53)=-acd75(36)+acd75(55)+acd75(54)+acd75(53)
      acd75(53)=acd75(33)*acd75(53)
      acd75(54)=acd75(39)*acd75(37)
      acd75(55)=acd75(41)*acd75(40)
      acd75(56)=acd75(43)*acd75(42)
      acd75(54)=-acd75(44)+acd75(56)+acd75(55)+acd75(54)
      acd75(54)=acd75(38)*acd75(54)
      acd75(55)=acd75(45)*acd75(7)
      acd75(56)=-acd75(47)*acd75(11)
      acd75(57)=acd75(48)*acd75(15)
      acd75(55)=-acd75(49)+acd75(57)+acd75(56)+acd75(55)
      acd75(55)=acd75(46)*acd75(55)
      acd75(56)=acd75(26)*acd75(8)
      acd75(57)=acd75(28)*acd75(12)
      acd75(56)=-acd75(29)+acd75(57)+acd75(56)
      acd75(56)=acd75(10)*acd75(56)
      acd75(57)=-acd75(25)*acd75(6)
      acd75(58)=-acd75(27)*acd75(8)
      acd75(59)=-acd75(30)*acd75(12)
      acd75(60)=-acd75(31)*acd75(14)
      brack=acd75(50)+acd75(51)+acd75(52)+acd75(53)+acd75(54)+acd75(55)+acd75(5&
      &6)+acd75(57)+acd75(58)+acd75(59)+acd75(60)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd75h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(91) :: acd75
      complex(ki) :: brack
      acd75(1)=k2(iv1)
      acd75(2)=dotproduct(qshift,spvae1k2)
      acd75(3)=abb75(10)
      acd75(4)=qshift(iv1)
      acd75(5)=abb75(30)
      acd75(6)=dotproduct(qshift,spval4e1)
      acd75(7)=abb75(15)
      acd75(8)=dotproduct(qshift,spvae1l4)
      acd75(9)=abb75(39)
      acd75(10)=dotproduct(qshift,spval5e1)
      acd75(11)=abb75(11)
      acd75(12)=dotproduct(qshift,spvae1e2)
      acd75(13)=abb75(27)
      acd75(14)=dotproduct(qshift,spvae2e1)
      acd75(15)=abb75(28)
      acd75(16)=abb75(21)
      acd75(17)=spvae1k2(iv1)
      acd75(18)=dotproduct(k2,qshift)
      acd75(19)=dotproduct(qshift,qshift)
      acd75(20)=abb75(29)
      acd75(21)=abb75(19)
      acd75(22)=abb75(22)
      acd75(23)=dotproduct(qshift,spvak2l4)
      acd75(24)=abb75(35)
      acd75(25)=dotproduct(qshift,spvak2e2)
      acd75(26)=abb75(23)
      acd75(27)=abb75(9)
      acd75(28)=spval4e1(iv1)
      acd75(29)=abb75(18)
      acd75(30)=spvae1l4(iv1)
      acd75(31)=abb75(32)
      acd75(32)=abb75(31)
      acd75(33)=spval5e1(iv1)
      acd75(34)=abb75(14)
      acd75(35)=abb75(13)
      acd75(36)=spvae1e2(iv1)
      acd75(37)=abb75(24)
      acd75(38)=spvae2e1(iv1)
      acd75(39)=abb75(20)
      acd75(40)=spvak2l4(iv1)
      acd75(41)=spval3k2(iv1)
      acd75(42)=dotproduct(qshift,spvae1l3)
      acd75(43)=spvae1l3(iv1)
      acd75(44)=dotproduct(qshift,spval3k2)
      acd75(45)=dotproduct(qshift,spval3l4)
      acd75(46)=dotproduct(qshift,spval3e2)
      acd75(47)=abb75(25)
      acd75(48)=spval3l4(iv1)
      acd75(49)=spval4k2(iv1)
      acd75(50)=dotproduct(qshift,spvak2e1)
      acd75(51)=abb75(34)
      acd75(52)=spvak2e1(iv1)
      acd75(53)=dotproduct(qshift,spval4k2)
      acd75(54)=dotproduct(qshift,spval5k2)
      acd75(55)=abb75(33)
      acd75(56)=dotproduct(qshift,spvae2k2)
      acd75(57)=abb75(26)
      acd75(58)=abb75(17)
      acd75(59)=spval4l3(iv1)
      acd75(60)=dotproduct(qshift,spval3e1)
      acd75(61)=spval3e1(iv1)
      acd75(62)=dotproduct(qshift,spval4l3)
      acd75(63)=dotproduct(qshift,spval5l3)
      acd75(64)=dotproduct(qshift,spvae2l3)
      acd75(65)=abb75(16)
      acd75(66)=spval5k2(iv1)
      acd75(67)=spval5l3(iv1)
      acd75(68)=spvae2k2(iv1)
      acd75(69)=spvak2e2(iv1)
      acd75(70)=spvae2l3(iv1)
      acd75(71)=spval3e2(iv1)
      acd75(72)=acd75(26)*acd75(69)
      acd75(73)=acd75(24)*acd75(40)
      acd75(74)=acd75(3)*acd75(1)
      acd75(75)=-acd75(38)*acd75(22)
      acd75(76)=-acd75(28)*acd75(20)
      acd75(77)=acd75(33)*acd75(21)
      acd75(78)=2.0_ki*acd75(4)
      acd75(79)=acd75(5)*acd75(78)
      acd75(72)=acd75(79)+acd75(77)+acd75(76)+acd75(75)+acd75(74)+acd75(72)+acd&
      &75(73)
      acd75(72)=acd75(2)*acd75(72)
      acd75(73)=acd75(26)*acd75(25)
      acd75(74)=acd75(24)*acd75(23)
      acd75(75)=-acd75(14)*acd75(22)
      acd75(76)=-acd75(6)*acd75(20)
      acd75(77)=acd75(3)*acd75(18)
      acd75(79)=acd75(10)*acd75(21)
      acd75(80)=acd75(19)*acd75(5)
      acd75(73)=acd75(80)+acd75(79)+acd75(77)+acd75(76)+acd75(75)+acd75(74)-acd&
      &75(27)+acd75(73)
      acd75(73)=acd75(17)*acd75(73)
      acd75(74)=acd75(15)*acd75(38)
      acd75(75)=-acd75(13)*acd75(36)
      acd75(76)=acd75(9)*acd75(30)
      acd75(77)=acd75(7)*acd75(28)
      acd75(79)=-acd75(33)*acd75(11)
      acd75(74)=acd75(79)+acd75(77)+acd75(76)+acd75(74)+acd75(75)
      acd75(74)=acd75(19)*acd75(74)
      acd75(75)=acd75(15)*acd75(14)
      acd75(76)=-acd75(13)*acd75(12)
      acd75(77)=-acd75(10)*acd75(11)
      acd75(79)=acd75(9)*acd75(8)
      acd75(80)=acd75(7)*acd75(6)
      acd75(75)=acd75(80)+acd75(79)+acd75(77)+acd75(76)+acd75(16)+acd75(75)
      acd75(75)=acd75(75)*acd75(78)
      acd75(76)=acd75(57)*acd75(68)
      acd75(77)=acd75(55)*acd75(66)
      acd75(78)=acd75(51)*acd75(49)
      acd75(76)=acd75(78)+acd75(76)+acd75(77)
      acd75(76)=acd75(50)*acd75(76)
      acd75(77)=acd75(57)*acd75(56)
      acd75(78)=acd75(55)*acd75(54)
      acd75(79)=acd75(51)*acd75(53)
      acd75(77)=acd75(79)+acd75(78)-acd75(58)+acd75(77)
      acd75(77)=acd75(52)*acd75(77)
      acd75(78)=acd75(60)*acd75(70)
      acd75(79)=acd75(61)*acd75(64)
      acd75(78)=acd75(78)+acd75(79)
      acd75(78)=acd75(15)*acd75(78)
      acd75(79)=-acd75(42)*acd75(71)
      acd75(80)=-acd75(43)*acd75(46)
      acd75(79)=acd75(79)+acd75(80)
      acd75(79)=acd75(13)*acd75(79)
      acd75(80)=-acd75(60)*acd75(67)
      acd75(81)=-acd75(61)*acd75(63)
      acd75(80)=acd75(80)+acd75(81)
      acd75(80)=acd75(11)*acd75(80)
      acd75(81)=acd75(36)*acd75(34)
      acd75(82)=acd75(30)*acd75(31)
      acd75(81)=acd75(81)+acd75(82)
      acd75(81)=acd75(10)*acd75(81)
      acd75(82)=acd75(42)*acd75(48)
      acd75(83)=acd75(43)*acd75(45)
      acd75(82)=acd75(82)+acd75(83)
      acd75(82)=acd75(9)*acd75(82)
      acd75(83)=acd75(60)*acd75(59)
      acd75(84)=acd75(61)*acd75(62)
      acd75(83)=acd75(83)+acd75(84)
      acd75(83)=acd75(7)*acd75(83)
      acd75(84)=acd75(42)*acd75(41)
      acd75(85)=acd75(43)*acd75(44)
      acd75(84)=acd75(84)+acd75(85)
      acd75(84)=acd75(5)*acd75(84)
      acd75(85)=acd75(12)*acd75(34)
      acd75(86)=acd75(8)*acd75(31)
      acd75(85)=acd75(86)-acd75(35)+acd75(85)
      acd75(85)=acd75(33)*acd75(85)
      acd75(86)=-acd75(38)*acd75(39)
      acd75(87)=-acd75(36)*acd75(37)
      acd75(88)=-acd75(30)*acd75(32)
      acd75(89)=-acd75(28)*acd75(29)
      acd75(90)=-acd75(61)*acd75(65)
      acd75(91)=-acd75(43)*acd75(47)
      brack=acd75(72)+acd75(73)+acd75(74)+acd75(75)+acd75(76)+acd75(77)+acd75(7&
      &8)+acd75(79)+acd75(80)+acd75(81)+acd75(82)+acd75(83)+acd75(84)+acd75(85)&
      &+acd75(86)+acd75(87)+acd75(88)+acd75(89)+acd75(90)+acd75(91)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd75h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(85) :: acd75
      complex(ki) :: brack
      acd75(1)=d(iv1,iv2)
      acd75(2)=dotproduct(qshift,spvae1k2)
      acd75(3)=abb75(30)
      acd75(4)=dotproduct(qshift,spval4e1)
      acd75(5)=abb75(15)
      acd75(6)=dotproduct(qshift,spvae1l4)
      acd75(7)=abb75(39)
      acd75(8)=dotproduct(qshift,spval5e1)
      acd75(9)=abb75(11)
      acd75(10)=dotproduct(qshift,spvae1e2)
      acd75(11)=abb75(27)
      acd75(12)=dotproduct(qshift,spvae2e1)
      acd75(13)=abb75(28)
      acd75(14)=abb75(21)
      acd75(15)=k2(iv1)
      acd75(16)=spvae1k2(iv2)
      acd75(17)=abb75(10)
      acd75(18)=k2(iv2)
      acd75(19)=spvae1k2(iv1)
      acd75(20)=qshift(iv1)
      acd75(21)=spval4e1(iv2)
      acd75(22)=spvae1l4(iv2)
      acd75(23)=spval5e1(iv2)
      acd75(24)=spvae1e2(iv2)
      acd75(25)=spvae2e1(iv2)
      acd75(26)=qshift(iv2)
      acd75(27)=spval4e1(iv1)
      acd75(28)=spvae1l4(iv1)
      acd75(29)=spval5e1(iv1)
      acd75(30)=spvae1e2(iv1)
      acd75(31)=spvae2e1(iv1)
      acd75(32)=abb75(29)
      acd75(33)=abb75(19)
      acd75(34)=abb75(22)
      acd75(35)=spvak2l4(iv2)
      acd75(36)=abb75(35)
      acd75(37)=spvak2e2(iv2)
      acd75(38)=abb75(23)
      acd75(39)=spvak2l4(iv1)
      acd75(40)=spvak2e2(iv1)
      acd75(41)=abb75(32)
      acd75(42)=abb75(14)
      acd75(43)=spval3k2(iv1)
      acd75(44)=spvae1l3(iv2)
      acd75(45)=spval3k2(iv2)
      acd75(46)=spvae1l3(iv1)
      acd75(47)=spval3l4(iv2)
      acd75(48)=spval3e2(iv2)
      acd75(49)=spval3l4(iv1)
      acd75(50)=spval3e2(iv1)
      acd75(51)=spval4k2(iv1)
      acd75(52)=spvak2e1(iv2)
      acd75(53)=abb75(34)
      acd75(54)=spval4k2(iv2)
      acd75(55)=spvak2e1(iv1)
      acd75(56)=spval5k2(iv2)
      acd75(57)=abb75(33)
      acd75(58)=spvae2k2(iv2)
      acd75(59)=abb75(26)
      acd75(60)=spval5k2(iv1)
      acd75(61)=spvae2k2(iv1)
      acd75(62)=spval4l3(iv1)
      acd75(63)=spval3e1(iv2)
      acd75(64)=spval4l3(iv2)
      acd75(65)=spval3e1(iv1)
      acd75(66)=spval5l3(iv2)
      acd75(67)=spvae2l3(iv2)
      acd75(68)=spval5l3(iv1)
      acd75(69)=spvae2l3(iv1)
      acd75(70)=acd75(38)*acd75(37)
      acd75(71)=acd75(36)*acd75(35)
      acd75(72)=-acd75(25)*acd75(34)
      acd75(73)=-acd75(21)*acd75(32)
      acd75(74)=acd75(17)*acd75(18)
      acd75(75)=acd75(23)*acd75(33)
      acd75(76)=2.0_ki*acd75(26)
      acd75(77)=acd75(3)*acd75(76)
      acd75(70)=acd75(77)+acd75(75)+acd75(74)+acd75(73)+acd75(72)+acd75(70)+acd&
      &75(71)
      acd75(70)=acd75(19)*acd75(70)
      acd75(71)=acd75(38)*acd75(40)
      acd75(72)=acd75(36)*acd75(39)
      acd75(73)=-acd75(31)*acd75(34)
      acd75(74)=-acd75(27)*acd75(32)
      acd75(75)=acd75(17)*acd75(15)
      acd75(77)=acd75(29)*acd75(33)
      acd75(78)=2.0_ki*acd75(20)
      acd75(79)=acd75(3)*acd75(78)
      acd75(71)=acd75(79)+acd75(77)+acd75(75)+acd75(74)+acd75(73)+acd75(71)+acd&
      &75(72)
      acd75(71)=acd75(16)*acd75(71)
      acd75(72)=acd75(13)*acd75(12)
      acd75(73)=-acd75(11)*acd75(10)
      acd75(74)=-acd75(9)*acd75(8)
      acd75(75)=acd75(7)*acd75(6)
      acd75(77)=acd75(5)*acd75(4)
      acd75(79)=acd75(3)*acd75(2)
      acd75(72)=acd75(79)+acd75(77)+acd75(75)+acd75(74)+acd75(73)+acd75(14)+acd&
      &75(72)
      acd75(72)=acd75(1)*acd75(72)
      acd75(73)=acd75(13)*acd75(31)
      acd75(74)=-acd75(11)*acd75(30)
      acd75(75)=-acd75(9)*acd75(29)
      acd75(77)=acd75(7)*acd75(28)
      acd75(79)=acd75(5)*acd75(27)
      acd75(73)=acd75(79)+acd75(77)+acd75(75)+acd75(73)+acd75(74)
      acd75(73)=acd75(73)*acd75(76)
      acd75(74)=acd75(13)*acd75(25)
      acd75(75)=-acd75(11)*acd75(24)
      acd75(76)=-acd75(9)*acd75(23)
      acd75(77)=acd75(7)*acd75(22)
      acd75(79)=acd75(5)*acd75(21)
      acd75(74)=acd75(79)+acd75(77)+acd75(76)+acd75(74)+acd75(75)
      acd75(74)=acd75(74)*acd75(78)
      acd75(75)=acd75(59)*acd75(58)
      acd75(76)=acd75(57)*acd75(56)
      acd75(77)=acd75(53)*acd75(54)
      acd75(75)=acd75(77)+acd75(75)+acd75(76)
      acd75(75)=acd75(55)*acd75(75)
      acd75(76)=acd75(59)*acd75(61)
      acd75(77)=acd75(57)*acd75(60)
      acd75(78)=acd75(53)*acd75(51)
      acd75(76)=acd75(78)+acd75(76)+acd75(77)
      acd75(76)=acd75(52)*acd75(76)
      acd75(77)=acd75(24)*acd75(42)
      acd75(78)=acd75(22)*acd75(41)
      acd75(77)=acd75(77)+acd75(78)
      acd75(77)=acd75(29)*acd75(77)
      acd75(78)=acd75(30)*acd75(42)
      acd75(79)=acd75(28)*acd75(41)
      acd75(78)=acd75(78)+acd75(79)
      acd75(78)=acd75(23)*acd75(78)
      acd75(79)=acd75(65)*acd75(67)
      acd75(80)=acd75(63)*acd75(69)
      acd75(79)=acd75(79)+acd75(80)
      acd75(79)=acd75(13)*acd75(79)
      acd75(80)=-acd75(46)*acd75(48)
      acd75(81)=-acd75(44)*acd75(50)
      acd75(80)=acd75(80)+acd75(81)
      acd75(80)=acd75(11)*acd75(80)
      acd75(81)=-acd75(65)*acd75(66)
      acd75(82)=-acd75(63)*acd75(68)
      acd75(81)=acd75(81)+acd75(82)
      acd75(81)=acd75(9)*acd75(81)
      acd75(82)=acd75(46)*acd75(47)
      acd75(83)=acd75(44)*acd75(49)
      acd75(82)=acd75(82)+acd75(83)
      acd75(82)=acd75(7)*acd75(82)
      acd75(83)=acd75(65)*acd75(64)
      acd75(84)=acd75(63)*acd75(62)
      acd75(83)=acd75(83)+acd75(84)
      acd75(83)=acd75(5)*acd75(83)
      acd75(84)=acd75(46)*acd75(45)
      acd75(85)=acd75(44)*acd75(43)
      acd75(84)=acd75(84)+acd75(85)
      acd75(84)=acd75(3)*acd75(84)
      brack=acd75(70)+acd75(71)+2.0_ki*acd75(72)+acd75(73)+acd75(74)+acd75(75)+&
      &acd75(76)+acd75(77)+acd75(78)+acd75(79)+acd75(80)+acd75(81)+acd75(82)+ac&
      &d75(83)+acd75(84)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd75h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd75
      complex(ki) :: brack
      acd75(1)=d(iv1,iv2)
      acd75(2)=spvae1k2(iv3)
      acd75(3)=abb75(30)
      acd75(4)=spval4e1(iv3)
      acd75(5)=abb75(15)
      acd75(6)=spvae1l4(iv3)
      acd75(7)=abb75(39)
      acd75(8)=spval5e1(iv3)
      acd75(9)=abb75(11)
      acd75(10)=spvae1e2(iv3)
      acd75(11)=abb75(27)
      acd75(12)=spvae2e1(iv3)
      acd75(13)=abb75(28)
      acd75(14)=d(iv1,iv3)
      acd75(15)=spvae1k2(iv2)
      acd75(16)=spval4e1(iv2)
      acd75(17)=spvae1l4(iv2)
      acd75(18)=spval5e1(iv2)
      acd75(19)=spvae1e2(iv2)
      acd75(20)=spvae2e1(iv2)
      acd75(21)=d(iv2,iv3)
      acd75(22)=spvae1k2(iv1)
      acd75(23)=spval4e1(iv1)
      acd75(24)=spvae1l4(iv1)
      acd75(25)=spval5e1(iv1)
      acd75(26)=spvae1e2(iv1)
      acd75(27)=spvae2e1(iv1)
      acd75(28)=acd75(2)*acd75(3)
      acd75(29)=acd75(4)*acd75(5)
      acd75(30)=acd75(6)*acd75(7)
      acd75(31)=-acd75(8)*acd75(9)
      acd75(32)=-acd75(10)*acd75(11)
      acd75(33)=acd75(12)*acd75(13)
      acd75(28)=acd75(33)+acd75(32)+acd75(31)+acd75(30)+acd75(28)+acd75(29)
      acd75(28)=acd75(1)*acd75(28)
      acd75(29)=acd75(15)*acd75(3)
      acd75(30)=acd75(16)*acd75(5)
      acd75(31)=acd75(17)*acd75(7)
      acd75(32)=-acd75(18)*acd75(9)
      acd75(33)=-acd75(19)*acd75(11)
      acd75(34)=acd75(20)*acd75(13)
      acd75(29)=acd75(34)+acd75(33)+acd75(32)+acd75(31)+acd75(30)+acd75(29)
      acd75(29)=acd75(14)*acd75(29)
      acd75(30)=acd75(22)*acd75(3)
      acd75(31)=acd75(23)*acd75(5)
      acd75(32)=acd75(24)*acd75(7)
      acd75(33)=-acd75(25)*acd75(9)
      acd75(34)=-acd75(26)*acd75(11)
      acd75(35)=acd75(27)*acd75(13)
      acd75(30)=acd75(35)+acd75(34)+acd75(33)+acd75(32)+acd75(31)+acd75(30)
      acd75(30)=acd75(21)*acd75(30)
      acd75(28)=acd75(30)+acd75(29)+acd75(28)
      brack=2.0_ki*acd75(28)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd75h4
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      integer, intent(in), optional :: i3
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k2-k3-k4
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
   end function derivative
!---#] function derivative:
end module     p2_gg_httbar_d75h4l1d
