module     p2_gg_httbar_d29h12l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity12d29h12l1d_qp.f90
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
      use p2_gg_httbar_abbrevd29h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(106) :: acd29
      complex(ki) :: brack
      acd29(1)=dotproduct(k2,qshift)
      acd29(2)=abb29(19)
      acd29(3)=dotproduct(qshift,qshift)
      acd29(4)=abb29(22)
      acd29(5)=dotproduct(qshift,spvak1k2)
      acd29(6)=abb29(12)
      acd29(7)=dotproduct(qshift,spvak1l3)
      acd29(8)=abb29(28)
      acd29(9)=dotproduct(qshift,spvak1l4)
      acd29(10)=abb29(41)
      acd29(11)=dotproduct(qshift,spvak2k1)
      acd29(12)=abb29(34)
      acd29(13)=dotproduct(qshift,spvak2l4)
      acd29(14)=abb29(33)
      acd29(15)=dotproduct(qshift,spvak2l5)
      acd29(16)=abb29(26)
      acd29(17)=dotproduct(qshift,spval3k1)
      acd29(18)=abb29(20)
      acd29(19)=dotproduct(qshift,spval3k2)
      acd29(20)=abb29(21)
      acd29(21)=dotproduct(qshift,spval3l4)
      acd29(22)=abb29(40)
      acd29(23)=dotproduct(qshift,spval3l5)
      acd29(24)=abb29(48)
      acd29(25)=dotproduct(qshift,spval4k1)
      acd29(26)=abb29(35)
      acd29(27)=dotproduct(qshift,spval4k2)
      acd29(28)=abb29(16)
      acd29(29)=dotproduct(qshift,spval4l3)
      acd29(30)=abb29(18)
      acd29(31)=dotproduct(qshift,spval4l5)
      acd29(32)=abb29(50)
      acd29(33)=dotproduct(qshift,spval5k2)
      acd29(34)=abb29(38)
      acd29(35)=dotproduct(qshift,spval5l3)
      acd29(36)=abb29(57)
      acd29(37)=dotproduct(qshift,spval5l4)
      acd29(38)=abb29(56)
      acd29(39)=dotproduct(qshift,spvak1e2)
      acd29(40)=abb29(58)
      acd29(41)=dotproduct(qshift,spvae2k1)
      acd29(42)=abb29(54)
      acd29(43)=dotproduct(qshift,spvak2e1)
      acd29(44)=abb29(52)
      acd29(45)=dotproduct(qshift,spvae1k2)
      acd29(46)=abb29(37)
      acd29(47)=dotproduct(qshift,spvak2e2)
      acd29(48)=abb29(30)
      acd29(49)=dotproduct(qshift,spvae2k2)
      acd29(50)=abb29(11)
      acd29(51)=dotproduct(qshift,spval3e1)
      acd29(52)=abb29(47)
      acd29(53)=dotproduct(qshift,spvae1l3)
      acd29(54)=abb29(46)
      acd29(55)=dotproduct(qshift,spval3e2)
      acd29(56)=abb29(43)
      acd29(57)=dotproduct(qshift,spvae2l3)
      acd29(58)=abb29(36)
      acd29(59)=dotproduct(qshift,spval4e1)
      acd29(60)=abb29(24)
      acd29(61)=dotproduct(qshift,spvae1l4)
      acd29(62)=abb29(70)
      acd29(63)=dotproduct(qshift,spval5e2)
      acd29(64)=abb29(32)
      acd29(65)=dotproduct(qshift,spvae2l5)
      acd29(66)=abb29(31)
      acd29(67)=dotproduct(qshift,spvae1e2)
      acd29(68)=abb29(23)
      acd29(69)=dotproduct(qshift,spvae2e1)
      acd29(70)=abb29(13)
      acd29(71)=abb29(15)
      acd29(72)=-acd29(2)*acd29(1)
      acd29(73)=acd29(4)*acd29(3)
      acd29(74)=-acd29(6)*acd29(5)
      acd29(75)=-acd29(8)*acd29(7)
      acd29(76)=-acd29(10)*acd29(9)
      acd29(77)=-acd29(12)*acd29(11)
      acd29(78)=-acd29(14)*acd29(13)
      acd29(79)=-acd29(16)*acd29(15)
      acd29(80)=-acd29(18)*acd29(17)
      acd29(81)=-acd29(20)*acd29(19)
      acd29(82)=-acd29(22)*acd29(21)
      acd29(83)=-acd29(24)*acd29(23)
      acd29(84)=-acd29(26)*acd29(25)
      acd29(85)=-acd29(28)*acd29(27)
      acd29(86)=-acd29(30)*acd29(29)
      acd29(87)=-acd29(32)*acd29(31)
      acd29(88)=-acd29(34)*acd29(33)
      acd29(89)=-acd29(36)*acd29(35)
      acd29(90)=-acd29(38)*acd29(37)
      acd29(91)=-acd29(40)*acd29(39)
      acd29(92)=-acd29(42)*acd29(41)
      acd29(93)=-acd29(44)*acd29(43)
      acd29(94)=-acd29(46)*acd29(45)
      acd29(95)=-acd29(48)*acd29(47)
      acd29(96)=-acd29(50)*acd29(49)
      acd29(97)=-acd29(52)*acd29(51)
      acd29(98)=-acd29(54)*acd29(53)
      acd29(99)=-acd29(56)*acd29(55)
      acd29(100)=-acd29(58)*acd29(57)
      acd29(101)=-acd29(60)*acd29(59)
      acd29(102)=acd29(62)*acd29(61)
      acd29(103)=-acd29(64)*acd29(63)
      acd29(104)=-acd29(66)*acd29(65)
      acd29(105)=-acd29(68)*acd29(67)
      acd29(106)=-acd29(70)*acd29(69)
      brack=acd29(71)+acd29(72)+acd29(73)+acd29(74)+acd29(75)+acd29(76)+acd29(7&
      &7)+acd29(78)+acd29(79)+acd29(80)+acd29(81)+acd29(82)+acd29(83)+acd29(84)&
      &+acd29(85)+acd29(86)+acd29(87)+acd29(88)+acd29(89)+acd29(90)+acd29(91)+a&
      &cd29(92)+acd29(93)+acd29(94)+acd29(95)+acd29(96)+acd29(97)+acd29(98)+acd&
      &29(99)+acd29(100)+acd29(101)+acd29(102)+acd29(103)+acd29(104)+acd29(105)&
      &+acd29(106)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd29h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(105) :: acd29
      complex(ki) :: brack
      acd29(1)=k2(iv1)
      acd29(2)=abb29(19)
      acd29(3)=qshift(iv1)
      acd29(4)=abb29(22)
      acd29(5)=spvak1k2(iv1)
      acd29(6)=abb29(12)
      acd29(7)=spvak1l3(iv1)
      acd29(8)=abb29(28)
      acd29(9)=spvak1l4(iv1)
      acd29(10)=abb29(41)
      acd29(11)=spvak2k1(iv1)
      acd29(12)=abb29(34)
      acd29(13)=spvak2l4(iv1)
      acd29(14)=abb29(33)
      acd29(15)=spvak2l5(iv1)
      acd29(16)=abb29(26)
      acd29(17)=spval3k1(iv1)
      acd29(18)=abb29(20)
      acd29(19)=spval3k2(iv1)
      acd29(20)=abb29(21)
      acd29(21)=spval3l4(iv1)
      acd29(22)=abb29(40)
      acd29(23)=spval3l5(iv1)
      acd29(24)=abb29(48)
      acd29(25)=spval4k1(iv1)
      acd29(26)=abb29(35)
      acd29(27)=spval4k2(iv1)
      acd29(28)=abb29(16)
      acd29(29)=spval4l3(iv1)
      acd29(30)=abb29(18)
      acd29(31)=spval4l5(iv1)
      acd29(32)=abb29(50)
      acd29(33)=spval5k2(iv1)
      acd29(34)=abb29(38)
      acd29(35)=spval5l3(iv1)
      acd29(36)=abb29(57)
      acd29(37)=spval5l4(iv1)
      acd29(38)=abb29(56)
      acd29(39)=spvak1e2(iv1)
      acd29(40)=abb29(58)
      acd29(41)=spvae2k1(iv1)
      acd29(42)=abb29(54)
      acd29(43)=spvak2e1(iv1)
      acd29(44)=abb29(52)
      acd29(45)=spvae1k2(iv1)
      acd29(46)=abb29(37)
      acd29(47)=spvak2e2(iv1)
      acd29(48)=abb29(30)
      acd29(49)=spvae2k2(iv1)
      acd29(50)=abb29(11)
      acd29(51)=spval3e1(iv1)
      acd29(52)=abb29(47)
      acd29(53)=spvae1l3(iv1)
      acd29(54)=abb29(46)
      acd29(55)=spval3e2(iv1)
      acd29(56)=abb29(43)
      acd29(57)=spvae2l3(iv1)
      acd29(58)=abb29(36)
      acd29(59)=spval4e1(iv1)
      acd29(60)=abb29(24)
      acd29(61)=spvae1l4(iv1)
      acd29(62)=abb29(70)
      acd29(63)=spval5e2(iv1)
      acd29(64)=abb29(32)
      acd29(65)=spvae2l5(iv1)
      acd29(66)=abb29(31)
      acd29(67)=spvae1e2(iv1)
      acd29(68)=abb29(23)
      acd29(69)=spvae2e1(iv1)
      acd29(70)=abb29(13)
      acd29(71)=acd29(2)*acd29(1)
      acd29(72)=acd29(4)*acd29(3)
      acd29(73)=acd29(6)*acd29(5)
      acd29(74)=acd29(8)*acd29(7)
      acd29(75)=acd29(10)*acd29(9)
      acd29(76)=acd29(12)*acd29(11)
      acd29(77)=acd29(14)*acd29(13)
      acd29(78)=acd29(16)*acd29(15)
      acd29(79)=acd29(18)*acd29(17)
      acd29(80)=acd29(20)*acd29(19)
      acd29(81)=acd29(22)*acd29(21)
      acd29(82)=acd29(24)*acd29(23)
      acd29(83)=acd29(26)*acd29(25)
      acd29(84)=acd29(28)*acd29(27)
      acd29(85)=acd29(30)*acd29(29)
      acd29(86)=acd29(32)*acd29(31)
      acd29(87)=acd29(34)*acd29(33)
      acd29(88)=acd29(36)*acd29(35)
      acd29(89)=acd29(38)*acd29(37)
      acd29(90)=acd29(40)*acd29(39)
      acd29(91)=acd29(42)*acd29(41)
      acd29(92)=acd29(44)*acd29(43)
      acd29(93)=acd29(46)*acd29(45)
      acd29(94)=acd29(48)*acd29(47)
      acd29(95)=acd29(50)*acd29(49)
      acd29(96)=acd29(52)*acd29(51)
      acd29(97)=acd29(54)*acd29(53)
      acd29(98)=acd29(56)*acd29(55)
      acd29(99)=acd29(58)*acd29(57)
      acd29(100)=acd29(60)*acd29(59)
      acd29(101)=-acd29(62)*acd29(61)
      acd29(102)=acd29(64)*acd29(63)
      acd29(103)=acd29(66)*acd29(65)
      acd29(104)=acd29(68)*acd29(67)
      acd29(105)=acd29(70)*acd29(69)
      brack=acd29(71)-2.0_ki*acd29(72)+acd29(73)+acd29(74)+acd29(75)+acd29(76)+&
      &acd29(77)+acd29(78)+acd29(79)+acd29(80)+acd29(81)+acd29(82)+acd29(83)+ac&
      &d29(84)+acd29(85)+acd29(86)+acd29(87)+acd29(88)+acd29(89)+acd29(90)+acd2&
      &9(91)+acd29(92)+acd29(93)+acd29(94)+acd29(95)+acd29(96)+acd29(97)+acd29(&
      &98)+acd29(99)+acd29(100)+acd29(101)+acd29(102)+acd29(103)+acd29(104)+acd&
      &29(105)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd29h12_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(3) :: acd29
      complex(ki) :: brack
      acd29(1)=d(iv1,iv2)
      acd29(2)=abb29(22)
      brack=2.0_ki*acd29(2)*acd29(1)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd29h12_qp
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
      qshift = k3+k4
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
end module     p2_gg_httbar_d29h12l1d_qp
