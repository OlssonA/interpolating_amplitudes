module     p2_gg_httbar_d148h0l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d148h0l1d_qp.f90
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
      use p2_gg_httbar_abbrevd148h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd148
      complex(ki) :: brack
      acd148(1)=abb148(40)
      brack=acd148(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd148h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(72) :: acd148
      complex(ki) :: brack
      acd148(1)=k2(iv1)
      acd148(2)=abb148(13)
      acd148(3)=l4(iv1)
      acd148(4)=abb148(38)
      acd148(5)=e2(iv1)
      acd148(6)=abb148(43)
      acd148(7)=spvak1l4(iv1)
      acd148(8)=abb148(21)
      acd148(9)=spvak2k1(iv1)
      acd148(10)=abb148(19)
      acd148(11)=spvak2l4(iv1)
      acd148(12)=abb148(11)
      acd148(13)=spvak2l5(iv1)
      acd148(14)=abb148(20)
      acd148(15)=spval4k1(iv1)
      acd148(16)=abb148(17)
      acd148(17)=spval4k2(iv1)
      acd148(18)=abb148(18)
      acd148(19)=spval4l5(iv1)
      acd148(20)=abb148(59)
      acd148(21)=spval5l4(iv1)
      acd148(22)=abb148(55)
      acd148(23)=spvak1e2(iv1)
      acd148(24)=abb148(50)
      acd148(25)=spvae2k1(iv1)
      acd148(26)=abb148(35)
      acd148(27)=spvak2e1(iv1)
      acd148(28)=abb148(39)
      acd148(29)=spvak2e2(iv1)
      acd148(30)=abb148(16)
      acd148(31)=spvae2k2(iv1)
      acd148(32)=abb148(14)
      acd148(33)=spval4e1(iv1)
      acd148(34)=abb148(41)
      acd148(35)=spvae1l4(iv1)
      acd148(36)=abb148(26)
      acd148(37)=spval4e2(iv1)
      acd148(38)=abb148(46)
      acd148(39)=spvae2l4(iv1)
      acd148(40)=abb148(91)
      acd148(41)=spval5e2(iv1)
      acd148(42)=abb148(145)
      acd148(43)=spvae2l5(iv1)
      acd148(44)=abb148(140)
      acd148(45)=spvae1e2(iv1)
      acd148(46)=abb148(47)
      acd148(47)=spvae2e1(iv1)
      acd148(48)=abb148(94)
      acd148(49)=acd148(2)*acd148(1)
      acd148(50)=acd148(4)*acd148(3)
      acd148(51)=acd148(6)*acd148(5)
      acd148(52)=acd148(8)*acd148(7)
      acd148(53)=acd148(10)*acd148(9)
      acd148(54)=acd148(12)*acd148(11)
      acd148(55)=acd148(14)*acd148(13)
      acd148(56)=acd148(16)*acd148(15)
      acd148(57)=acd148(18)*acd148(17)
      acd148(58)=acd148(20)*acd148(19)
      acd148(59)=acd148(22)*acd148(21)
      acd148(60)=acd148(24)*acd148(23)
      acd148(61)=acd148(26)*acd148(25)
      acd148(62)=acd148(28)*acd148(27)
      acd148(63)=acd148(30)*acd148(29)
      acd148(64)=acd148(32)*acd148(31)
      acd148(65)=acd148(34)*acd148(33)
      acd148(66)=acd148(36)*acd148(35)
      acd148(67)=acd148(38)*acd148(37)
      acd148(68)=acd148(40)*acd148(39)
      acd148(69)=acd148(42)*acd148(41)
      acd148(70)=acd148(44)*acd148(43)
      acd148(71)=acd148(46)*acd148(45)
      acd148(72)=acd148(48)*acd148(47)
      brack=acd148(49)+acd148(50)+acd148(51)+acd148(52)+acd148(53)+acd148(54)+a&
      &cd148(55)+acd148(56)+acd148(57)+acd148(58)+acd148(59)+acd148(60)+acd148(&
      &61)+acd148(62)+acd148(63)+acd148(64)+acd148(65)+acd148(66)+acd148(67)+ac&
      &d148(68)+acd148(69)+acd148(70)+acd148(71)+acd148(72)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd148h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(41) :: acd148
      complex(ki) :: brack
      acd148(1)=d(iv1,iv2)
      acd148(2)=abb148(105)
      acd148(3)=k2(iv1)
      acd148(4)=e2(iv2)
      acd148(5)=abb148(65)
      acd148(6)=k2(iv2)
      acd148(7)=e2(iv1)
      acd148(8)=l4(iv1)
      acd148(9)=abb148(57)
      acd148(10)=l4(iv2)
      acd148(11)=spvak1k2(iv2)
      acd148(12)=abb148(22)
      acd148(13)=spval4k1(iv2)
      acd148(14)=abb148(12)
      acd148(15)=spval4k2(iv2)
      acd148(16)=abb148(15)
      acd148(17)=spval4l5(iv2)
      acd148(18)=abb148(30)
      acd148(19)=spval5k2(iv2)
      acd148(20)=abb148(69)
      acd148(21)=spvae1k2(iv2)
      acd148(22)=abb148(24)
      acd148(23)=spval4e1(iv2)
      acd148(24)=abb148(86)
      acd148(25)=spvak1k2(iv1)
      acd148(26)=spval4k1(iv1)
      acd148(27)=spval4k2(iv1)
      acd148(28)=spval4l5(iv1)
      acd148(29)=spval5k2(iv1)
      acd148(30)=spvae1k2(iv1)
      acd148(31)=spval4e1(iv1)
      acd148(32)=acd148(3)*acd148(5)
      acd148(33)=acd148(8)*acd148(9)
      acd148(34)=acd148(25)*acd148(12)
      acd148(35)=acd148(26)*acd148(14)
      acd148(36)=acd148(27)*acd148(16)
      acd148(37)=acd148(28)*acd148(18)
      acd148(38)=acd148(29)*acd148(20)
      acd148(39)=acd148(30)*acd148(22)
      acd148(40)=acd148(31)*acd148(24)
      acd148(32)=acd148(40)+acd148(39)+acd148(38)+acd148(37)+acd148(36)+acd148(&
      &35)+acd148(34)+acd148(33)+acd148(32)
      acd148(32)=acd148(4)*acd148(32)
      acd148(33)=acd148(6)*acd148(5)
      acd148(34)=acd148(10)*acd148(9)
      acd148(35)=acd148(11)*acd148(12)
      acd148(36)=acd148(13)*acd148(14)
      acd148(37)=acd148(15)*acd148(16)
      acd148(38)=acd148(17)*acd148(18)
      acd148(39)=acd148(19)*acd148(20)
      acd148(40)=acd148(21)*acd148(22)
      acd148(41)=acd148(23)*acd148(24)
      acd148(33)=acd148(41)+acd148(40)+acd148(39)+acd148(38)+acd148(37)+acd148(&
      &36)+acd148(35)+acd148(34)+acd148(33)
      acd148(33)=acd148(7)*acd148(33)
      acd148(34)=acd148(2)*acd148(1)
      brack=acd148(32)+acd148(33)+2.0_ki*acd148(34)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd148h0_qp
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
end module     p2_gg_httbar_d148h0l1d_qp
