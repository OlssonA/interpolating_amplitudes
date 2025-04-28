module     p2_gg_httbar_d163h0l1d
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d163h0l1d.f90
   ! generator: buildfortran_d.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond, d => metric_tensor
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
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd163h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(88) :: acd163
      complex(ki) :: brack
      acd163(1)=dotproduct(k2,qshift)
      acd163(2)=dotproduct(e1,qshift)
      acd163(3)=abb163(62)
      acd163(4)=abb163(50)
      acd163(5)=dotproduct(l5,qshift)
      acd163(6)=abb163(45)
      acd163(7)=abb163(18)
      acd163(8)=dotproduct(qshift,spvak1k2)
      acd163(9)=abb163(17)
      acd163(10)=dotproduct(qshift,spval4k2)
      acd163(11)=abb163(30)
      acd163(12)=dotproduct(qshift,spval5k1)
      acd163(13)=abb163(26)
      acd163(14)=dotproduct(qshift,spval5k2)
      acd163(15)=abb163(23)
      acd163(16)=dotproduct(qshift,spval5l4)
      acd163(17)=abb163(27)
      acd163(18)=dotproduct(qshift,spvae2k2)
      acd163(19)=abb163(56)
      acd163(20)=dotproduct(qshift,spval5e2)
      acd163(21)=abb163(80)
      acd163(22)=abb163(15)
      acd163(23)=dotproduct(qshift,qshift)
      acd163(24)=abb163(83)
      acd163(25)=abb163(25)
      acd163(26)=abb163(20)
      acd163(27)=abb163(29)
      acd163(28)=abb163(54)
      acd163(29)=dotproduct(qshift,spvak1l5)
      acd163(30)=abb163(21)
      acd163(31)=dotproduct(qshift,spvak2k1)
      acd163(32)=abb163(19)
      acd163(33)=dotproduct(qshift,spvak2l4)
      acd163(34)=abb163(166)
      acd163(35)=dotproduct(qshift,spvak2l5)
      acd163(36)=abb163(11)
      acd163(37)=dotproduct(qshift,spval4l5)
      acd163(38)=abb163(24)
      acd163(39)=dotproduct(qshift,spvak1e1)
      acd163(40)=abb163(22)
      acd163(41)=dotproduct(qshift,spvae1k1)
      acd163(42)=abb163(28)
      acd163(43)=dotproduct(qshift,spvak2e1)
      acd163(44)=abb163(48)
      acd163(45)=dotproduct(qshift,spvae1k2)
      acd163(46)=abb163(16)
      acd163(47)=dotproduct(qshift,spvak2e2)
      acd163(48)=abb163(12)
      acd163(49)=dotproduct(qshift,spval4e1)
      acd163(50)=abb163(46)
      acd163(51)=dotproduct(qshift,spvae1l4)
      acd163(52)=abb163(154)
      acd163(53)=dotproduct(qshift,spval5e1)
      acd163(54)=abb163(33)
      acd163(55)=dotproduct(qshift,spvae1l5)
      acd163(56)=abb163(84)
      acd163(57)=dotproduct(qshift,spvae2l5)
      acd163(58)=abb163(31)
      acd163(59)=dotproduct(qshift,spvae1e2)
      acd163(60)=abb163(14)
      acd163(61)=dotproduct(qshift,spvae2e1)
      acd163(62)=abb163(52)
      acd163(63)=abb163(13)
      acd163(64)=acd163(3)*acd163(1)
      acd163(65)=acd163(6)*acd163(5)
      acd163(66)=acd163(9)*acd163(8)
      acd163(67)=acd163(11)*acd163(10)
      acd163(68)=acd163(13)*acd163(12)
      acd163(69)=acd163(15)*acd163(14)
      acd163(70)=acd163(17)*acd163(16)
      acd163(71)=acd163(19)*acd163(18)
      acd163(72)=acd163(21)*acd163(20)
      acd163(64)=-acd163(22)+acd163(72)+acd163(71)+acd163(70)+acd163(69)+acd163&
      &(68)+acd163(67)+acd163(66)+acd163(65)+acd163(64)
      acd163(64)=acd163(2)*acd163(64)
      acd163(65)=-acd163(4)*acd163(1)
      acd163(66)=-acd163(7)*acd163(5)
      acd163(67)=acd163(24)*acd163(23)
      acd163(68)=-acd163(25)*acd163(12)
      acd163(69)=-acd163(26)*acd163(14)
      acd163(70)=-acd163(27)*acd163(16)
      acd163(71)=-acd163(28)*acd163(20)
      acd163(72)=-acd163(30)*acd163(29)
      acd163(73)=-acd163(32)*acd163(31)
      acd163(74)=-acd163(34)*acd163(33)
      acd163(75)=-acd163(36)*acd163(35)
      acd163(76)=-acd163(38)*acd163(37)
      acd163(77)=-acd163(40)*acd163(39)
      acd163(78)=-acd163(42)*acd163(41)
      acd163(79)=-acd163(44)*acd163(43)
      acd163(80)=-acd163(46)*acd163(45)
      acd163(81)=-acd163(48)*acd163(47)
      acd163(82)=-acd163(50)*acd163(49)
      acd163(83)=acd163(52)*acd163(51)
      acd163(84)=-acd163(54)*acd163(53)
      acd163(85)=-acd163(56)*acd163(55)
      acd163(86)=-acd163(58)*acd163(57)
      acd163(87)=-acd163(60)*acd163(59)
      acd163(88)=-acd163(62)*acd163(61)
      brack=acd163(63)+acd163(64)+acd163(65)+acd163(66)+acd163(67)+acd163(68)+a&
      &cd163(69)+acd163(70)+acd163(71)+acd163(72)+acd163(73)+acd163(74)+acd163(&
      &75)+acd163(76)+acd163(77)+acd163(78)+acd163(79)+acd163(80)+acd163(81)+ac&
      &d163(82)+acd163(83)+acd163(84)+acd163(85)+acd163(86)+acd163(87)+acd163(8&
      &8)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd163h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(98) :: acd163
      complex(ki) :: brack
      acd163(1)=k2(iv1)
      acd163(2)=dotproduct(e1,qshift)
      acd163(3)=abb163(62)
      acd163(4)=abb163(50)
      acd163(5)=l5(iv1)
      acd163(6)=abb163(45)
      acd163(7)=abb163(18)
      acd163(8)=e1(iv1)
      acd163(9)=dotproduct(k2,qshift)
      acd163(10)=dotproduct(l5,qshift)
      acd163(11)=dotproduct(qshift,spvak1k2)
      acd163(12)=abb163(17)
      acd163(13)=dotproduct(qshift,spval4k2)
      acd163(14)=abb163(30)
      acd163(15)=dotproduct(qshift,spval5k1)
      acd163(16)=abb163(26)
      acd163(17)=dotproduct(qshift,spval5k2)
      acd163(18)=abb163(23)
      acd163(19)=dotproduct(qshift,spval5l4)
      acd163(20)=abb163(27)
      acd163(21)=dotproduct(qshift,spvae2k2)
      acd163(22)=abb163(56)
      acd163(23)=dotproduct(qshift,spval5e2)
      acd163(24)=abb163(80)
      acd163(25)=abb163(15)
      acd163(26)=qshift(iv1)
      acd163(27)=abb163(83)
      acd163(28)=spvak1k2(iv1)
      acd163(29)=spval4k2(iv1)
      acd163(30)=spval5k1(iv1)
      acd163(31)=abb163(25)
      acd163(32)=spval5k2(iv1)
      acd163(33)=abb163(20)
      acd163(34)=spval5l4(iv1)
      acd163(35)=abb163(29)
      acd163(36)=spvae2k2(iv1)
      acd163(37)=spval5e2(iv1)
      acd163(38)=abb163(54)
      acd163(39)=spvak1l5(iv1)
      acd163(40)=abb163(21)
      acd163(41)=spvak2k1(iv1)
      acd163(42)=abb163(19)
      acd163(43)=spvak2l4(iv1)
      acd163(44)=abb163(166)
      acd163(45)=spvak2l5(iv1)
      acd163(46)=abb163(11)
      acd163(47)=spval4l5(iv1)
      acd163(48)=abb163(24)
      acd163(49)=spvak1e1(iv1)
      acd163(50)=abb163(22)
      acd163(51)=spvae1k1(iv1)
      acd163(52)=abb163(28)
      acd163(53)=spvak2e1(iv1)
      acd163(54)=abb163(48)
      acd163(55)=spvae1k2(iv1)
      acd163(56)=abb163(16)
      acd163(57)=spvak2e2(iv1)
      acd163(58)=abb163(12)
      acd163(59)=spval4e1(iv1)
      acd163(60)=abb163(46)
      acd163(61)=spvae1l4(iv1)
      acd163(62)=abb163(154)
      acd163(63)=spval5e1(iv1)
      acd163(64)=abb163(33)
      acd163(65)=spvae1l5(iv1)
      acd163(66)=abb163(84)
      acd163(67)=spvae2l5(iv1)
      acd163(68)=abb163(31)
      acd163(69)=spvae1e2(iv1)
      acd163(70)=abb163(14)
      acd163(71)=spvae2e1(iv1)
      acd163(72)=abb163(52)
      acd163(73)=acd163(3)*acd163(1)
      acd163(74)=acd163(6)*acd163(5)
      acd163(75)=acd163(30)*acd163(16)
      acd163(76)=acd163(32)*acd163(18)
      acd163(77)=acd163(34)*acd163(20)
      acd163(78)=acd163(37)*acd163(24)
      acd163(79)=acd163(28)*acd163(12)
      acd163(80)=acd163(29)*acd163(14)
      acd163(81)=acd163(36)*acd163(22)
      acd163(73)=acd163(81)+acd163(80)+acd163(79)+acd163(78)+acd163(77)+acd163(&
      &76)+acd163(75)+acd163(73)+acd163(74)
      acd163(73)=acd163(2)*acd163(73)
      acd163(74)=acd163(9)*acd163(3)
      acd163(75)=acd163(10)*acd163(6)
      acd163(76)=acd163(11)*acd163(12)
      acd163(77)=acd163(13)*acd163(14)
      acd163(78)=acd163(15)*acd163(16)
      acd163(79)=acd163(17)*acd163(18)
      acd163(80)=acd163(19)*acd163(20)
      acd163(81)=acd163(21)*acd163(22)
      acd163(82)=acd163(23)*acd163(24)
      acd163(74)=-acd163(25)+acd163(82)+acd163(81)+acd163(80)+acd163(79)+acd163&
      &(78)+acd163(77)+acd163(76)+acd163(75)+acd163(74)
      acd163(74)=acd163(8)*acd163(74)
      acd163(75)=-acd163(4)*acd163(1)
      acd163(76)=-acd163(7)*acd163(5)
      acd163(77)=acd163(27)*acd163(26)
      acd163(78)=-acd163(31)*acd163(30)
      acd163(79)=-acd163(33)*acd163(32)
      acd163(80)=-acd163(35)*acd163(34)
      acd163(81)=-acd163(38)*acd163(37)
      acd163(82)=-acd163(40)*acd163(39)
      acd163(83)=-acd163(42)*acd163(41)
      acd163(84)=-acd163(44)*acd163(43)
      acd163(85)=-acd163(46)*acd163(45)
      acd163(86)=-acd163(48)*acd163(47)
      acd163(87)=-acd163(50)*acd163(49)
      acd163(88)=-acd163(52)*acd163(51)
      acd163(89)=-acd163(54)*acd163(53)
      acd163(90)=-acd163(56)*acd163(55)
      acd163(91)=-acd163(58)*acd163(57)
      acd163(92)=-acd163(60)*acd163(59)
      acd163(93)=acd163(62)*acd163(61)
      acd163(94)=-acd163(64)*acd163(63)
      acd163(95)=-acd163(66)*acd163(65)
      acd163(96)=-acd163(68)*acd163(67)
      acd163(97)=-acd163(70)*acd163(69)
      acd163(98)=-acd163(72)*acd163(71)
      brack=acd163(73)+acd163(74)+acd163(75)+acd163(76)+2.0_ki*acd163(77)+acd16&
      &3(78)+acd163(79)+acd163(80)+acd163(81)+acd163(82)+acd163(83)+acd163(84)+&
      &acd163(85)+acd163(86)+acd163(87)+acd163(88)+acd163(89)+acd163(90)+acd163&
      &(91)+acd163(92)+acd163(93)+acd163(94)+acd163(95)+acd163(96)+acd163(97)+a&
      &cd163(98)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd163h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(41) :: acd163
      complex(ki) :: brack
      acd163(1)=d(iv1,iv2)
      acd163(2)=abb163(83)
      acd163(3)=k2(iv1)
      acd163(4)=e1(iv2)
      acd163(5)=abb163(62)
      acd163(6)=k2(iv2)
      acd163(7)=e1(iv1)
      acd163(8)=l5(iv1)
      acd163(9)=abb163(45)
      acd163(10)=l5(iv2)
      acd163(11)=spvak1k2(iv2)
      acd163(12)=abb163(17)
      acd163(13)=spval4k2(iv2)
      acd163(14)=abb163(30)
      acd163(15)=spval5k1(iv2)
      acd163(16)=abb163(26)
      acd163(17)=spval5k2(iv2)
      acd163(18)=abb163(23)
      acd163(19)=spval5l4(iv2)
      acd163(20)=abb163(27)
      acd163(21)=spvae2k2(iv2)
      acd163(22)=abb163(56)
      acd163(23)=spval5e2(iv2)
      acd163(24)=abb163(80)
      acd163(25)=spvak1k2(iv1)
      acd163(26)=spval4k2(iv1)
      acd163(27)=spval5k1(iv1)
      acd163(28)=spval5k2(iv1)
      acd163(29)=spval5l4(iv1)
      acd163(30)=spvae2k2(iv1)
      acd163(31)=spval5e2(iv1)
      acd163(32)=acd163(3)*acd163(5)
      acd163(33)=acd163(8)*acd163(9)
      acd163(34)=acd163(25)*acd163(12)
      acd163(35)=acd163(26)*acd163(14)
      acd163(36)=acd163(27)*acd163(16)
      acd163(37)=acd163(28)*acd163(18)
      acd163(38)=acd163(29)*acd163(20)
      acd163(39)=acd163(30)*acd163(22)
      acd163(40)=acd163(31)*acd163(24)
      acd163(32)=acd163(40)+acd163(39)+acd163(38)+acd163(37)+acd163(36)+acd163(&
      &35)+acd163(34)+acd163(33)+acd163(32)
      acd163(32)=acd163(4)*acd163(32)
      acd163(33)=acd163(6)*acd163(5)
      acd163(34)=acd163(10)*acd163(9)
      acd163(35)=acd163(11)*acd163(12)
      acd163(36)=acd163(13)*acd163(14)
      acd163(37)=acd163(15)*acd163(16)
      acd163(38)=acd163(17)*acd163(18)
      acd163(39)=acd163(19)*acd163(20)
      acd163(40)=acd163(21)*acd163(22)
      acd163(41)=acd163(23)*acd163(24)
      acd163(33)=acd163(41)+acd163(40)+acd163(39)+acd163(38)+acd163(37)+acd163(&
      &36)+acd163(35)+acd163(34)+acd163(33)
      acd163(33)=acd163(7)*acd163(33)
      acd163(34)=acd163(2)*acd163(1)
      brack=acd163(32)+acd163(33)+2.0_ki*acd163(34)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd163h0
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
      qshift = -k3-k4-k5
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
end module     p2_gg_httbar_d163h0l1d
