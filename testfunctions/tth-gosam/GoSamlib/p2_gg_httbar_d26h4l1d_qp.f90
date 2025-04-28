module     p2_gg_httbar_d26h4l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d26h4l1d_qp.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd26h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(109) :: acd26
      complex(ki) :: brack
      acd26(1)=dotproduct(k2,qshift)
      acd26(2)=abb26(14)
      acd26(3)=dotproduct(qshift,qshift)
      acd26(4)=abb26(17)
      acd26(5)=dotproduct(qshift,spvak1k2)
      acd26(6)=abb26(37)
      acd26(7)=dotproduct(qshift,spvak1l3)
      acd26(8)=abb26(18)
      acd26(9)=dotproduct(qshift,spvak1l5)
      acd26(10)=abb26(33)
      acd26(11)=dotproduct(qshift,spvak2k1)
      acd26(12)=abb26(25)
      acd26(13)=dotproduct(qshift,spvak2l3)
      acd26(14)=abb26(15)
      acd26(15)=dotproduct(qshift,spvak2l4)
      acd26(16)=abb26(28)
      acd26(17)=dotproduct(qshift,spvak2l5)
      acd26(18)=abb26(36)
      acd26(19)=dotproduct(qshift,spval3k1)
      acd26(20)=abb26(27)
      acd26(21)=dotproduct(qshift,spval3k2)
      acd26(22)=abb26(20)
      acd26(23)=dotproduct(qshift,spval3l4)
      acd26(24)=abb26(45)
      acd26(25)=dotproduct(qshift,spval3l5)
      acd26(26)=abb26(44)
      acd26(27)=dotproduct(qshift,spval4k2)
      acd26(28)=abb26(24)
      acd26(29)=dotproduct(qshift,spval4l3)
      acd26(30)=abb26(21)
      acd26(31)=dotproduct(qshift,spval4l5)
      acd26(32)=abb26(34)
      acd26(33)=dotproduct(qshift,spval5k1)
      acd26(34)=abb26(29)
      acd26(35)=dotproduct(qshift,spval5k2)
      acd26(36)=abb26(16)
      acd26(37)=dotproduct(qshift,spval5l3)
      acd26(38)=abb26(43)
      acd26(39)=dotproduct(qshift,spval5l4)
      acd26(40)=abb26(40)
      acd26(41)=dotproduct(qshift,spvak1e2)
      acd26(42)=abb26(32)
      acd26(43)=dotproduct(qshift,spvae2k1)
      acd26(44)=abb26(30)
      acd26(45)=dotproduct(qshift,spvak2e1)
      acd26(46)=abb26(12)
      acd26(47)=dotproduct(qshift,spvae1k2)
      acd26(48)=abb26(72)
      acd26(49)=dotproduct(qshift,spvak2e2)
      acd26(50)=abb26(19)
      acd26(51)=dotproduct(qshift,spvae2k2)
      acd26(52)=abb26(11)
      acd26(53)=dotproduct(qshift,spval3e1)
      acd26(54)=abb26(73)
      acd26(55)=dotproduct(qshift,spvae1l3)
      acd26(56)=abb26(57)
      acd26(57)=dotproduct(qshift,spval3e2)
      acd26(58)=abb26(42)
      acd26(59)=dotproduct(qshift,spvae2l3)
      acd26(60)=abb26(39)
      acd26(61)=dotproduct(qshift,spval4e2)
      acd26(62)=abb26(35)
      acd26(63)=dotproduct(qshift,spvae2l4)
      acd26(64)=abb26(31)
      acd26(65)=dotproduct(qshift,spval5e1)
      acd26(66)=abb26(26)
      acd26(67)=dotproduct(qshift,spvae1l5)
      acd26(68)=abb26(23)
      acd26(69)=dotproduct(qshift,spvae1e2)
      acd26(70)=abb26(52)
      acd26(71)=dotproduct(qshift,spvae2e1)
      acd26(72)=abb26(22)
      acd26(73)=abb26(13)
      acd26(74)=-acd26(2)*acd26(1)
      acd26(75)=acd26(4)*acd26(3)
      acd26(76)=-acd26(6)*acd26(5)
      acd26(77)=-acd26(8)*acd26(7)
      acd26(78)=-acd26(10)*acd26(9)
      acd26(79)=-acd26(12)*acd26(11)
      acd26(80)=-acd26(14)*acd26(13)
      acd26(81)=-acd26(16)*acd26(15)
      acd26(82)=-acd26(18)*acd26(17)
      acd26(83)=-acd26(20)*acd26(19)
      acd26(84)=-acd26(22)*acd26(21)
      acd26(85)=-acd26(24)*acd26(23)
      acd26(86)=-acd26(26)*acd26(25)
      acd26(87)=-acd26(28)*acd26(27)
      acd26(88)=-acd26(30)*acd26(29)
      acd26(89)=-acd26(32)*acd26(31)
      acd26(90)=-acd26(34)*acd26(33)
      acd26(91)=-acd26(36)*acd26(35)
      acd26(92)=-acd26(38)*acd26(37)
      acd26(93)=-acd26(40)*acd26(39)
      acd26(94)=-acd26(42)*acd26(41)
      acd26(95)=-acd26(44)*acd26(43)
      acd26(96)=-acd26(46)*acd26(45)
      acd26(97)=-acd26(48)*acd26(47)
      acd26(98)=-acd26(50)*acd26(49)
      acd26(99)=-acd26(52)*acd26(51)
      acd26(100)=acd26(54)*acd26(53)
      acd26(101)=-acd26(56)*acd26(55)
      acd26(102)=-acd26(58)*acd26(57)
      acd26(103)=-acd26(60)*acd26(59)
      acd26(104)=-acd26(62)*acd26(61)
      acd26(105)=-acd26(64)*acd26(63)
      acd26(106)=-acd26(66)*acd26(65)
      acd26(107)=-acd26(68)*acd26(67)
      acd26(108)=acd26(70)*acd26(69)
      acd26(109)=-acd26(72)*acd26(71)
      brack=acd26(73)+acd26(74)+acd26(75)+acd26(76)+acd26(77)+acd26(78)+acd26(7&
      &9)+acd26(80)+acd26(81)+acd26(82)+acd26(83)+acd26(84)+acd26(85)+acd26(86)&
      &+acd26(87)+acd26(88)+acd26(89)+acd26(90)+acd26(91)+acd26(92)+acd26(93)+a&
      &cd26(94)+acd26(95)+acd26(96)+acd26(97)+acd26(98)+acd26(99)+acd26(100)+ac&
      &d26(101)+acd26(102)+acd26(103)+acd26(104)+acd26(105)+acd26(106)+acd26(10&
      &7)+acd26(108)+acd26(109)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd26h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(108) :: acd26
      complex(ki) :: brack
      acd26(1)=k2(iv1)
      acd26(2)=abb26(14)
      acd26(3)=qshift(iv1)
      acd26(4)=abb26(17)
      acd26(5)=spvak1k2(iv1)
      acd26(6)=abb26(37)
      acd26(7)=spvak1l3(iv1)
      acd26(8)=abb26(18)
      acd26(9)=spvak1l5(iv1)
      acd26(10)=abb26(33)
      acd26(11)=spvak2k1(iv1)
      acd26(12)=abb26(25)
      acd26(13)=spvak2l3(iv1)
      acd26(14)=abb26(15)
      acd26(15)=spvak2l4(iv1)
      acd26(16)=abb26(28)
      acd26(17)=spvak2l5(iv1)
      acd26(18)=abb26(36)
      acd26(19)=spval3k1(iv1)
      acd26(20)=abb26(27)
      acd26(21)=spval3k2(iv1)
      acd26(22)=abb26(20)
      acd26(23)=spval3l4(iv1)
      acd26(24)=abb26(45)
      acd26(25)=spval3l5(iv1)
      acd26(26)=abb26(44)
      acd26(27)=spval4k2(iv1)
      acd26(28)=abb26(24)
      acd26(29)=spval4l3(iv1)
      acd26(30)=abb26(21)
      acd26(31)=spval4l5(iv1)
      acd26(32)=abb26(34)
      acd26(33)=spval5k1(iv1)
      acd26(34)=abb26(29)
      acd26(35)=spval5k2(iv1)
      acd26(36)=abb26(16)
      acd26(37)=spval5l3(iv1)
      acd26(38)=abb26(43)
      acd26(39)=spval5l4(iv1)
      acd26(40)=abb26(40)
      acd26(41)=spvak1e2(iv1)
      acd26(42)=abb26(32)
      acd26(43)=spvae2k1(iv1)
      acd26(44)=abb26(30)
      acd26(45)=spvak2e1(iv1)
      acd26(46)=abb26(12)
      acd26(47)=spvae1k2(iv1)
      acd26(48)=abb26(72)
      acd26(49)=spvak2e2(iv1)
      acd26(50)=abb26(19)
      acd26(51)=spvae2k2(iv1)
      acd26(52)=abb26(11)
      acd26(53)=spval3e1(iv1)
      acd26(54)=abb26(73)
      acd26(55)=spvae1l3(iv1)
      acd26(56)=abb26(57)
      acd26(57)=spval3e2(iv1)
      acd26(58)=abb26(42)
      acd26(59)=spvae2l3(iv1)
      acd26(60)=abb26(39)
      acd26(61)=spval4e2(iv1)
      acd26(62)=abb26(35)
      acd26(63)=spvae2l4(iv1)
      acd26(64)=abb26(31)
      acd26(65)=spval5e1(iv1)
      acd26(66)=abb26(26)
      acd26(67)=spvae1l5(iv1)
      acd26(68)=abb26(23)
      acd26(69)=spvae1e2(iv1)
      acd26(70)=abb26(52)
      acd26(71)=spvae2e1(iv1)
      acd26(72)=abb26(22)
      acd26(73)=acd26(2)*acd26(1)
      acd26(74)=acd26(4)*acd26(3)
      acd26(75)=acd26(6)*acd26(5)
      acd26(76)=acd26(8)*acd26(7)
      acd26(77)=acd26(10)*acd26(9)
      acd26(78)=acd26(12)*acd26(11)
      acd26(79)=acd26(14)*acd26(13)
      acd26(80)=acd26(16)*acd26(15)
      acd26(81)=acd26(18)*acd26(17)
      acd26(82)=acd26(20)*acd26(19)
      acd26(83)=acd26(22)*acd26(21)
      acd26(84)=acd26(24)*acd26(23)
      acd26(85)=acd26(26)*acd26(25)
      acd26(86)=acd26(28)*acd26(27)
      acd26(87)=acd26(30)*acd26(29)
      acd26(88)=acd26(32)*acd26(31)
      acd26(89)=acd26(34)*acd26(33)
      acd26(90)=acd26(36)*acd26(35)
      acd26(91)=acd26(38)*acd26(37)
      acd26(92)=acd26(40)*acd26(39)
      acd26(93)=acd26(42)*acd26(41)
      acd26(94)=acd26(44)*acd26(43)
      acd26(95)=acd26(46)*acd26(45)
      acd26(96)=acd26(48)*acd26(47)
      acd26(97)=acd26(50)*acd26(49)
      acd26(98)=acd26(52)*acd26(51)
      acd26(99)=-acd26(54)*acd26(53)
      acd26(100)=acd26(56)*acd26(55)
      acd26(101)=acd26(58)*acd26(57)
      acd26(102)=acd26(60)*acd26(59)
      acd26(103)=acd26(62)*acd26(61)
      acd26(104)=acd26(64)*acd26(63)
      acd26(105)=acd26(66)*acd26(65)
      acd26(106)=acd26(68)*acd26(67)
      acd26(107)=-acd26(70)*acd26(69)
      acd26(108)=acd26(72)*acd26(71)
      brack=acd26(73)-2.0_ki*acd26(74)+acd26(75)+acd26(76)+acd26(77)+acd26(78)+&
      &acd26(79)+acd26(80)+acd26(81)+acd26(82)+acd26(83)+acd26(84)+acd26(85)+ac&
      &d26(86)+acd26(87)+acd26(88)+acd26(89)+acd26(90)+acd26(91)+acd26(92)+acd2&
      &6(93)+acd26(94)+acd26(95)+acd26(96)+acd26(97)+acd26(98)+acd26(99)+acd26(&
      &100)+acd26(101)+acd26(102)+acd26(103)+acd26(104)+acd26(105)+acd26(106)+a&
      &cd26(107)+acd26(108)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd26h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(3) :: acd26
      complex(ki) :: brack
      acd26(1)=d(iv1,iv2)
      acd26(2)=abb26(17)
      brack=2.0_ki*acd26(2)*acd26(1)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd26h4_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k3+k5
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
   end function derivative
!---#] function derivative:
end module     p2_gg_httbar_d26h4l1d_qp
