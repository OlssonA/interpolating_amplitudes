module     p2_gg_httbar_d74h8l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d74h8l1d_qp.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd74h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd74
      complex(ki) :: brack
      acd74(1)=abb74(16)
      brack=acd74(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd74h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(111) :: acd74
      complex(ki) :: brack
      acd74(1)=k2(iv1)
      acd74(2)=abb74(12)
      acd74(3)=l4(iv1)
      acd74(4)=abb74(49)
      acd74(5)=e2(iv1)
      acd74(6)=abb74(26)
      acd74(7)=spvak1k2(iv1)
      acd74(8)=abb74(51)
      acd74(9)=spvak1l3(iv1)
      acd74(10)=abb74(47)
      acd74(11)=spvak1l4(iv1)
      acd74(12)=abb74(11)
      acd74(13)=spvak2k1(iv1)
      acd74(14)=abb74(10)
      acd74(15)=spvak2l4(iv1)
      acd74(16)=abb74(45)
      acd74(17)=spvak2l5(iv1)
      acd74(18)=abb74(42)
      acd74(19)=spval3k1(iv1)
      acd74(20)=abb74(24)
      acd74(21)=spval3k2(iv1)
      acd74(22)=abb74(21)
      acd74(23)=spval3l5(iv1)
      acd74(24)=abb74(29)
      acd74(25)=spval4k1(iv1)
      acd74(26)=abb74(20)
      acd74(27)=spval4k2(iv1)
      acd74(28)=abb74(13)
      acd74(29)=spval4l3(iv1)
      acd74(30)=abb74(36)
      acd74(31)=spval4l5(iv1)
      acd74(32)=abb74(112)
      acd74(33)=spval5k2(iv1)
      acd74(34)=abb74(40)
      acd74(35)=spval5l3(iv1)
      acd74(36)=abb74(89)
      acd74(37)=spval5l4(iv1)
      acd74(38)=abb74(70)
      acd74(39)=spvak1e2(iv1)
      acd74(40)=abb74(15)
      acd74(41)=spvae2k1(iv1)
      acd74(42)=abb74(43)
      acd74(43)=spvak2e1(iv1)
      acd74(44)=abb74(25)
      acd74(45)=spvae1k2(iv1)
      acd74(46)=abb74(113)
      acd74(47)=spvak2e2(iv1)
      acd74(48)=abb74(87)
      acd74(49)=spvae2k2(iv1)
      acd74(50)=abb74(9)
      acd74(51)=spval3e1(iv1)
      acd74(52)=abb74(60)
      acd74(53)=spvae1l3(iv1)
      acd74(54)=abb74(91)
      acd74(55)=spval3e2(iv1)
      acd74(56)=abb74(30)
      acd74(57)=spvae2l3(iv1)
      acd74(58)=abb74(78)
      acd74(59)=spval4e1(iv1)
      acd74(60)=abb74(97)
      acd74(61)=spvae1l4(iv1)
      acd74(62)=abb74(57)
      acd74(63)=spval4e2(iv1)
      acd74(64)=abb74(14)
      acd74(65)=spvae2l4(iv1)
      acd74(66)=abb74(80)
      acd74(67)=spval5e2(iv1)
      acd74(68)=abb74(75)
      acd74(69)=spvae2l5(iv1)
      acd74(70)=abb74(48)
      acd74(71)=spvae1e2(iv1)
      acd74(72)=abb74(33)
      acd74(73)=spvae2e1(iv1)
      acd74(74)=abb74(34)
      acd74(75)=-acd74(2)*acd74(1)
      acd74(76)=-acd74(4)*acd74(3)
      acd74(77)=-acd74(6)*acd74(5)
      acd74(78)=-acd74(8)*acd74(7)
      acd74(79)=-acd74(10)*acd74(9)
      acd74(80)=-acd74(12)*acd74(11)
      acd74(81)=-acd74(14)*acd74(13)
      acd74(82)=-acd74(16)*acd74(15)
      acd74(83)=-acd74(18)*acd74(17)
      acd74(84)=-acd74(20)*acd74(19)
      acd74(85)=-acd74(22)*acd74(21)
      acd74(86)=-acd74(24)*acd74(23)
      acd74(87)=-acd74(26)*acd74(25)
      acd74(88)=-acd74(28)*acd74(27)
      acd74(89)=-acd74(30)*acd74(29)
      acd74(90)=-acd74(32)*acd74(31)
      acd74(91)=-acd74(34)*acd74(33)
      acd74(92)=-acd74(36)*acd74(35)
      acd74(93)=-acd74(38)*acd74(37)
      acd74(94)=-acd74(40)*acd74(39)
      acd74(95)=-acd74(42)*acd74(41)
      acd74(96)=-acd74(44)*acd74(43)
      acd74(97)=-acd74(46)*acd74(45)
      acd74(98)=-acd74(48)*acd74(47)
      acd74(99)=-acd74(50)*acd74(49)
      acd74(100)=-acd74(52)*acd74(51)
      acd74(101)=-acd74(54)*acd74(53)
      acd74(102)=-acd74(56)*acd74(55)
      acd74(103)=-acd74(58)*acd74(57)
      acd74(104)=-acd74(60)*acd74(59)
      acd74(105)=-acd74(62)*acd74(61)
      acd74(106)=-acd74(64)*acd74(63)
      acd74(107)=-acd74(66)*acd74(65)
      acd74(108)=-acd74(68)*acd74(67)
      acd74(109)=-acd74(70)*acd74(69)
      acd74(110)=-acd74(72)*acd74(71)
      acd74(111)=-acd74(74)*acd74(73)
      brack=acd74(75)+acd74(76)+acd74(77)+acd74(78)+acd74(79)+acd74(80)+acd74(8&
      &1)+acd74(82)+acd74(83)+acd74(84)+acd74(85)+acd74(86)+acd74(87)+acd74(88)&
      &+acd74(89)+acd74(90)+acd74(91)+acd74(92)+acd74(93)+acd74(94)+acd74(95)+a&
      &cd74(96)+acd74(97)+acd74(98)+acd74(99)+acd74(100)+acd74(101)+acd74(102)+&
      &acd74(103)+acd74(104)+acd74(105)+acd74(106)+acd74(107)+acd74(108)+acd74(&
      &109)+acd74(110)+acd74(111)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd74h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(114) :: acd74
      complex(ki) :: brack
      acd74(1)=d(iv1,iv2)
      acd74(2)=abb74(22)
      acd74(3)=k2(iv1)
      acd74(4)=e2(iv2)
      acd74(5)=abb74(17)
      acd74(6)=spvae2k2(iv2)
      acd74(7)=abb74(35)
      acd74(8)=k2(iv2)
      acd74(9)=e2(iv1)
      acd74(10)=spvae2k2(iv1)
      acd74(11)=l4(iv1)
      acd74(12)=abb74(103)
      acd74(13)=spval4e2(iv2)
      acd74(14)=abb74(31)
      acd74(15)=l4(iv2)
      acd74(16)=spval4e2(iv1)
      acd74(17)=spvak1k2(iv2)
      acd74(18)=abb74(52)
      acd74(19)=spvak1l3(iv2)
      acd74(20)=abb74(50)
      acd74(21)=spval3k1(iv2)
      acd74(22)=abb74(41)
      acd74(23)=spval3k2(iv2)
      acd74(24)=abb74(37)
      acd74(25)=spval3l5(iv2)
      acd74(26)=abb74(32)
      acd74(27)=spval4k1(iv2)
      acd74(28)=abb74(23)
      acd74(29)=spval4k2(iv2)
      acd74(30)=abb74(18)
      acd74(31)=spval4l3(iv2)
      acd74(32)=abb74(28)
      acd74(33)=spval4l5(iv2)
      acd74(34)=abb74(114)
      acd74(35)=spval5k2(iv2)
      acd74(36)=abb74(108)
      acd74(37)=spval5l3(iv2)
      acd74(38)=abb74(100)
      acd74(39)=spvae1k2(iv2)
      acd74(40)=abb74(105)
      acd74(41)=spval3e1(iv2)
      acd74(42)=abb74(126)
      acd74(43)=spvae1l3(iv2)
      acd74(44)=abb74(67)
      acd74(45)=spval4e1(iv2)
      acd74(46)=abb74(102)
      acd74(47)=spvak1k2(iv1)
      acd74(48)=spvak1l3(iv1)
      acd74(49)=spval3k1(iv1)
      acd74(50)=spval3k2(iv1)
      acd74(51)=spval3l5(iv1)
      acd74(52)=spval4k1(iv1)
      acd74(53)=spval4k2(iv1)
      acd74(54)=spval4l3(iv1)
      acd74(55)=spval4l5(iv1)
      acd74(56)=spval5k2(iv1)
      acd74(57)=spval5l3(iv1)
      acd74(58)=spvae1k2(iv1)
      acd74(59)=spval3e1(iv1)
      acd74(60)=spvae1l3(iv1)
      acd74(61)=spval4e1(iv1)
      acd74(62)=spvak2k1(iv2)
      acd74(63)=abb74(46)
      acd74(64)=spvak2l5(iv2)
      acd74(65)=abb74(44)
      acd74(66)=spvak2e1(iv2)
      acd74(67)=abb74(62)
      acd74(68)=spvak2k1(iv1)
      acd74(69)=spvak2l5(iv1)
      acd74(70)=spvak2e1(iv1)
      acd74(71)=spvak1l4(iv2)
      acd74(72)=abb74(84)
      acd74(73)=spval5l4(iv2)
      acd74(74)=abb74(83)
      acd74(75)=spvae1l4(iv2)
      acd74(76)=abb74(85)
      acd74(77)=spvak1l4(iv1)
      acd74(78)=spval5l4(iv1)
      acd74(79)=spvae1l4(iv1)
      acd74(80)=spvak2e2(iv2)
      acd74(81)=abb74(54)
      acd74(82)=spvak2e2(iv1)
      acd74(83)=spval3e2(iv2)
      acd74(84)=spval3e2(iv1)
      acd74(85)=spvae2l3(iv2)
      acd74(86)=abb74(65)
      acd74(87)=spvae2l3(iv1)
      acd74(88)=abb74(39)
      acd74(89)=abb74(56)
      acd74(90)=spvae2l4(iv2)
      acd74(91)=spvae2l4(iv1)
      acd74(92)=abb74(19)
      acd74(93)=abb74(81)
      acd74(94)=abb74(59)
      acd74(95)=abb74(122)
      acd74(96)=abb74(38)
      acd74(97)=acd74(45)*acd74(46)
      acd74(98)=acd74(43)*acd74(44)
      acd74(99)=acd74(41)*acd74(42)
      acd74(100)=acd74(39)*acd74(40)
      acd74(101)=acd74(37)*acd74(38)
      acd74(102)=acd74(35)*acd74(36)
      acd74(103)=acd74(33)*acd74(34)
      acd74(104)=acd74(32)*acd74(31)
      acd74(105)=acd74(27)*acd74(28)
      acd74(106)=acd74(25)*acd74(26)
      acd74(107)=acd74(23)*acd74(24)
      acd74(108)=acd74(21)*acd74(22)
      acd74(109)=acd74(19)*acd74(20)
      acd74(110)=acd74(17)*acd74(18)
      acd74(111)=acd74(12)*acd74(15)
      acd74(112)=acd74(5)*acd74(8)
      acd74(113)=acd74(29)*acd74(30)
      acd74(97)=acd74(113)+acd74(112)+acd74(111)+acd74(110)+acd74(109)+acd74(10&
      &8)+acd74(107)+acd74(106)+acd74(105)+acd74(104)+acd74(103)+acd74(102)+acd&
      &74(101)+acd74(100)+acd74(99)+acd74(97)+acd74(98)
      acd74(97)=acd74(9)*acd74(97)
      acd74(98)=acd74(46)*acd74(61)
      acd74(99)=acd74(44)*acd74(60)
      acd74(100)=acd74(42)*acd74(59)
      acd74(101)=acd74(40)*acd74(58)
      acd74(102)=acd74(38)*acd74(57)
      acd74(103)=acd74(36)*acd74(56)
      acd74(104)=acd74(34)*acd74(55)
      acd74(105)=acd74(32)*acd74(54)
      acd74(106)=acd74(28)*acd74(52)
      acd74(107)=acd74(26)*acd74(51)
      acd74(108)=acd74(24)*acd74(50)
      acd74(109)=acd74(22)*acd74(49)
      acd74(110)=acd74(20)*acd74(48)
      acd74(111)=acd74(18)*acd74(47)
      acd74(112)=acd74(11)*acd74(12)
      acd74(113)=acd74(3)*acd74(5)
      acd74(114)=acd74(53)*acd74(30)
      acd74(98)=acd74(114)+acd74(113)+acd74(112)+acd74(111)+acd74(110)+acd74(10&
      &9)+acd74(108)+acd74(107)+acd74(106)+acd74(105)+acd74(104)+acd74(103)+acd&
      &74(102)+acd74(101)+acd74(100)+acd74(98)+acd74(99)
      acd74(98)=acd74(4)*acd74(98)
      acd74(99)=acd74(50)*acd74(88)
      acd74(100)=-acd74(96)*acd74(59)
      acd74(101)=acd74(89)*acd74(51)
      acd74(102)=-acd74(86)*acd74(49)
      acd74(99)=acd74(102)+acd74(101)+acd74(99)+acd74(100)
      acd74(99)=acd74(85)*acd74(99)
      acd74(100)=acd74(39)*acd74(95)
      acd74(101)=acd74(35)*acd74(94)
      acd74(102)=acd74(17)*acd74(81)
      acd74(103)=acd74(29)*acd74(92)
      acd74(100)=acd74(103)+acd74(102)+acd74(100)+acd74(101)
      acd74(100)=acd74(82)*acd74(100)
      acd74(101)=acd74(58)*acd74(95)
      acd74(102)=acd74(56)*acd74(94)
      acd74(103)=acd74(47)*acd74(81)
      acd74(104)=acd74(53)*acd74(92)
      acd74(101)=acd74(104)+acd74(103)+acd74(101)+acd74(102)
      acd74(101)=acd74(80)*acd74(101)
      acd74(102)=acd74(14)*acd74(15)
      acd74(103)=acd74(76)*acd74(75)
      acd74(104)=acd74(74)*acd74(73)
      acd74(105)=-acd74(72)*acd74(71)
      acd74(102)=acd74(105)+acd74(104)+acd74(102)+acd74(103)
      acd74(102)=acd74(16)*acd74(102)
      acd74(103)=acd74(11)*acd74(14)
      acd74(104)=acd74(76)*acd74(79)
      acd74(105)=acd74(74)*acd74(78)
      acd74(106)=-acd74(72)*acd74(77)
      acd74(103)=acd74(106)+acd74(105)+acd74(103)+acd74(104)
      acd74(103)=acd74(13)*acd74(103)
      acd74(104)=acd74(67)*acd74(66)
      acd74(105)=acd74(65)*acd74(64)
      acd74(106)=acd74(63)*acd74(62)
      acd74(107)=acd74(7)*acd74(8)
      acd74(104)=acd74(107)+acd74(106)+acd74(104)+acd74(105)
      acd74(104)=acd74(10)*acd74(104)
      acd74(105)=acd74(67)*acd74(70)
      acd74(106)=acd74(65)*acd74(69)
      acd74(107)=acd74(63)*acd74(68)
      acd74(108)=acd74(3)*acd74(7)
      acd74(105)=acd74(108)+acd74(107)+acd74(105)+acd74(106)
      acd74(105)=acd74(6)*acd74(105)
      acd74(106)=acd74(23)*acd74(88)
      acd74(107)=-acd74(96)*acd74(41)
      acd74(108)=acd74(89)*acd74(25)
      acd74(106)=acd74(108)+acd74(106)+acd74(107)
      acd74(106)=acd74(87)*acd74(106)
      acd74(107)=-acd74(91)*acd74(27)
      acd74(108)=-acd74(90)*acd74(52)
      acd74(109)=-acd74(87)*acd74(21)
      acd74(107)=acd74(109)+acd74(107)+acd74(108)
      acd74(107)=acd74(86)*acd74(107)
      acd74(108)=-acd74(29)*acd74(93)
      acd74(109)=-acd74(96)*acd74(45)
      acd74(108)=acd74(108)+acd74(109)
      acd74(108)=acd74(91)*acd74(108)
      acd74(109)=-acd74(53)*acd74(93)
      acd74(110)=-acd74(96)*acd74(61)
      acd74(109)=acd74(109)+acd74(110)
      acd74(109)=acd74(90)*acd74(109)
      acd74(110)=acd74(91)*acd74(33)
      acd74(111)=acd74(90)*acd74(55)
      acd74(110)=acd74(110)+acd74(111)
      acd74(110)=acd74(89)*acd74(110)
      acd74(111)=acd74(84)*acd74(43)
      acd74(112)=acd74(83)*acd74(60)
      acd74(111)=acd74(111)+acd74(112)
      acd74(111)=acd74(76)*acd74(111)
      acd74(112)=acd74(84)*acd74(37)
      acd74(113)=acd74(83)*acd74(57)
      acd74(112)=acd74(112)+acd74(113)
      acd74(112)=acd74(74)*acd74(112)
      acd74(113)=-acd74(84)*acd74(19)
      acd74(114)=-acd74(83)*acd74(48)
      acd74(113)=acd74(113)+acd74(114)
      acd74(113)=acd74(72)*acd74(113)
      acd74(114)=acd74(1)*acd74(2)
      brack=acd74(97)+acd74(98)+acd74(99)+acd74(100)+acd74(101)+acd74(102)+acd7&
      &4(103)+acd74(104)+acd74(105)+acd74(106)+acd74(107)+acd74(108)+acd74(109)&
      &+acd74(110)+acd74(111)+acd74(112)+acd74(113)+2.0_ki*acd74(114)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd74h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(50) :: acd74
      complex(ki) :: brack
      acd74(1)=d(iv1,iv2)
      acd74(2)=e2(iv3)
      acd74(3)=abb74(27)
      acd74(4)=spvak1e2(iv3)
      acd74(5)=abb74(84)
      acd74(6)=spvae2k1(iv3)
      acd74(7)=abb74(65)
      acd74(8)=spvae2k2(iv3)
      acd74(9)=abb74(81)
      acd74(10)=spval4e2(iv3)
      acd74(11)=abb74(53)
      acd74(12)=spval5e2(iv3)
      acd74(13)=abb74(83)
      acd74(14)=spvae2l5(iv3)
      acd74(15)=abb74(56)
      acd74(16)=spvae1e2(iv3)
      acd74(17)=abb74(85)
      acd74(18)=spvae2e1(iv3)
      acd74(19)=abb74(38)
      acd74(20)=d(iv1,iv3)
      acd74(21)=e2(iv2)
      acd74(22)=spvak1e2(iv2)
      acd74(23)=spvae2k1(iv2)
      acd74(24)=spvae2k2(iv2)
      acd74(25)=spval4e2(iv2)
      acd74(26)=spval5e2(iv2)
      acd74(27)=spvae2l5(iv2)
      acd74(28)=spvae1e2(iv2)
      acd74(29)=spvae2e1(iv2)
      acd74(30)=d(iv2,iv3)
      acd74(31)=e2(iv1)
      acd74(32)=spvak1e2(iv1)
      acd74(33)=spvae2k1(iv1)
      acd74(34)=spvae2k2(iv1)
      acd74(35)=spval4e2(iv1)
      acd74(36)=spval5e2(iv1)
      acd74(37)=spvae2l5(iv1)
      acd74(38)=spvae1e2(iv1)
      acd74(39)=spvae2e1(iv1)
      acd74(40)=-acd74(2)*acd74(3)
      acd74(41)=acd74(4)*acd74(5)
      acd74(42)=acd74(6)*acd74(7)
      acd74(43)=acd74(8)*acd74(9)
      acd74(44)=-acd74(10)*acd74(11)
      acd74(45)=-acd74(12)*acd74(13)
      acd74(46)=-acd74(14)*acd74(15)
      acd74(47)=-acd74(16)*acd74(17)
      acd74(48)=acd74(18)*acd74(19)
      acd74(40)=acd74(48)+acd74(47)+acd74(46)+acd74(45)+acd74(44)+acd74(43)+acd&
      &74(42)+acd74(40)+acd74(41)
      acd74(40)=acd74(1)*acd74(40)
      acd74(41)=-acd74(21)*acd74(3)
      acd74(42)=acd74(22)*acd74(5)
      acd74(43)=acd74(23)*acd74(7)
      acd74(44)=acd74(24)*acd74(9)
      acd74(45)=-acd74(25)*acd74(11)
      acd74(46)=-acd74(26)*acd74(13)
      acd74(47)=-acd74(27)*acd74(15)
      acd74(48)=-acd74(28)*acd74(17)
      acd74(49)=acd74(29)*acd74(19)
      acd74(41)=acd74(49)+acd74(48)+acd74(47)+acd74(46)+acd74(45)+acd74(44)+acd&
      &74(43)+acd74(42)+acd74(41)
      acd74(41)=acd74(20)*acd74(41)
      acd74(42)=-acd74(31)*acd74(3)
      acd74(43)=acd74(32)*acd74(5)
      acd74(44)=acd74(33)*acd74(7)
      acd74(45)=acd74(34)*acd74(9)
      acd74(46)=-acd74(35)*acd74(11)
      acd74(47)=-acd74(36)*acd74(13)
      acd74(48)=-acd74(37)*acd74(15)
      acd74(49)=-acd74(38)*acd74(17)
      acd74(50)=acd74(39)*acd74(19)
      acd74(42)=acd74(50)+acd74(49)+acd74(48)+acd74(47)+acd74(46)+acd74(45)+acd&
      &74(44)+acd74(43)+acd74(42)
      acd74(42)=acd74(30)*acd74(42)
      acd74(40)=acd74(42)+acd74(41)+acd74(40)
      brack=2.0_ki*acd74(40)
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd74h8_qp
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
      qshift = 0
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
end module     p2_gg_httbar_d74h8l1d_qp
