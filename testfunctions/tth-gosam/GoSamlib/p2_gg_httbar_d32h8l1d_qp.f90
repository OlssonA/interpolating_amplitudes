module     p2_gg_httbar_d32h8l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d32h8l1d_qp.f90
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
      use p2_gg_httbar_abbrevd32h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd32
      complex(ki) :: brack
      acd32(1)=dotproduct(qshift,spvak2e1)
      acd32(2)=dotproduct(qshift,spvae1l3)
      acd32(3)=abb32(32)
      acd32(4)=dotproduct(qshift,spvae1l4)
      acd32(5)=abb32(11)
      acd32(6)=dotproduct(qshift,spvae1l5)
      acd32(7)=abb32(21)
      acd32(8)=dotproduct(qshift,spvae1e2)
      acd32(9)=abb32(20)
      acd32(10)=abb32(10)
      acd32(11)=dotproduct(qshift,spval4e1)
      acd32(12)=abb32(25)
      acd32(13)=dotproduct(qshift,spvae2e1)
      acd32(14)=abb32(24)
      acd32(15)=abb32(17)
      acd32(16)=dotproduct(qshift,spval3e1)
      acd32(17)=abb32(46)
      acd32(18)=abb32(27)
      acd32(19)=abb32(29)
      acd32(20)=abb32(26)
      acd32(21)=abb32(19)
      acd32(22)=abb32(48)
      acd32(23)=abb32(15)
      acd32(24)=abb32(16)
      acd32(25)=abb32(28)
      acd32(26)=abb32(14)
      acd32(27)=abb32(13)
      acd32(28)=-acd32(3)*acd32(2)
      acd32(29)=-acd32(5)*acd32(4)
      acd32(30)=acd32(7)*acd32(6)
      acd32(31)=-acd32(9)*acd32(8)
      acd32(28)=-acd32(10)+acd32(31)+acd32(30)+acd32(28)+acd32(29)
      acd32(28)=acd32(1)*acd32(28)
      acd32(29)=acd32(12)*acd32(2)
      acd32(30)=acd32(19)*acd32(6)
      acd32(29)=-acd32(25)+acd32(30)+acd32(29)
      acd32(29)=acd32(11)*acd32(29)
      acd32(30)=-acd32(14)*acd32(2)
      acd32(31)=-acd32(20)*acd32(6)
      acd32(30)=-acd32(26)+acd32(31)+acd32(30)
      acd32(30)=acd32(13)*acd32(30)
      acd32(31)=-acd32(17)*acd32(4)
      acd32(32)=-acd32(22)*acd32(8)
      acd32(31)=-acd32(24)+acd32(32)+acd32(31)
      acd32(31)=acd32(16)*acd32(31)
      acd32(32)=-acd32(15)*acd32(2)
      acd32(33)=-acd32(18)*acd32(4)
      acd32(34)=-acd32(21)*acd32(6)
      acd32(35)=-acd32(23)*acd32(8)
      brack=acd32(27)+acd32(28)+acd32(29)+acd32(30)+acd32(31)+acd32(32)+acd32(3&
      &3)+acd32(34)+acd32(35)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd32h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(45) :: acd32
      complex(ki) :: brack
      acd32(1)=spvak2e1(iv1)
      acd32(2)=dotproduct(qshift,spvae1l3)
      acd32(3)=abb32(32)
      acd32(4)=dotproduct(qshift,spvae1l4)
      acd32(5)=abb32(11)
      acd32(6)=dotproduct(qshift,spvae1l5)
      acd32(7)=abb32(21)
      acd32(8)=dotproduct(qshift,spvae1e2)
      acd32(9)=abb32(20)
      acd32(10)=abb32(10)
      acd32(11)=spvae1l3(iv1)
      acd32(12)=dotproduct(qshift,spvak2e1)
      acd32(13)=dotproduct(qshift,spval4e1)
      acd32(14)=abb32(25)
      acd32(15)=dotproduct(qshift,spvae2e1)
      acd32(16)=abb32(24)
      acd32(17)=abb32(17)
      acd32(18)=spvae1l4(iv1)
      acd32(19)=dotproduct(qshift,spval3e1)
      acd32(20)=abb32(46)
      acd32(21)=abb32(27)
      acd32(22)=spvae1l5(iv1)
      acd32(23)=abb32(29)
      acd32(24)=abb32(26)
      acd32(25)=abb32(19)
      acd32(26)=spvae1e2(iv1)
      acd32(27)=abb32(48)
      acd32(28)=abb32(15)
      acd32(29)=spval3e1(iv1)
      acd32(30)=abb32(16)
      acd32(31)=spval4e1(iv1)
      acd32(32)=abb32(28)
      acd32(33)=spvae2e1(iv1)
      acd32(34)=abb32(14)
      acd32(35)=-acd32(8)*acd32(9)
      acd32(36)=-acd32(4)*acd32(5)
      acd32(37)=acd32(6)*acd32(7)
      acd32(38)=-acd32(2)*acd32(3)
      acd32(35)=acd32(38)+acd32(37)+acd32(36)-acd32(10)+acd32(35)
      acd32(35)=acd32(1)*acd32(35)
      acd32(36)=-acd32(26)*acd32(9)
      acd32(37)=-acd32(18)*acd32(5)
      acd32(38)=acd32(22)*acd32(7)
      acd32(36)=acd32(38)+acd32(36)+acd32(37)
      acd32(36)=acd32(12)*acd32(36)
      acd32(37)=-acd32(15)*acd32(16)
      acd32(38)=acd32(13)*acd32(14)
      acd32(39)=-acd32(12)*acd32(3)
      acd32(37)=acd32(39)+acd32(38)-acd32(17)+acd32(37)
      acd32(37)=acd32(11)*acd32(37)
      acd32(38)=-acd32(8)*acd32(27)
      acd32(39)=-acd32(4)*acd32(20)
      acd32(38)=acd32(39)-acd32(30)+acd32(38)
      acd32(38)=acd32(29)*acd32(38)
      acd32(39)=-acd32(33)*acd32(24)
      acd32(40)=acd32(31)*acd32(23)
      acd32(39)=acd32(39)+acd32(40)
      acd32(39)=acd32(6)*acd32(39)
      acd32(40)=-acd32(33)*acd32(16)
      acd32(41)=acd32(31)*acd32(14)
      acd32(40)=acd32(40)+acd32(41)
      acd32(40)=acd32(2)*acd32(40)
      acd32(41)=-acd32(15)*acd32(24)
      acd32(42)=acd32(13)*acd32(23)
      acd32(41)=acd32(42)-acd32(25)+acd32(41)
      acd32(41)=acd32(22)*acd32(41)
      acd32(42)=-acd32(33)*acd32(34)
      acd32(43)=-acd32(31)*acd32(32)
      acd32(44)=-acd32(19)*acd32(27)
      acd32(44)=-acd32(28)+acd32(44)
      acd32(44)=acd32(26)*acd32(44)
      acd32(45)=-acd32(19)*acd32(20)
      acd32(45)=-acd32(21)+acd32(45)
      acd32(45)=acd32(18)*acd32(45)
      brack=acd32(35)+acd32(36)+acd32(37)+acd32(38)+acd32(39)+acd32(40)+acd32(4&
      &1)+acd32(42)+acd32(43)+acd32(44)+acd32(45)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd32h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd32
      complex(ki) :: brack
      acd32(1)=spvak2e1(iv1)
      acd32(2)=spvae1l3(iv2)
      acd32(3)=abb32(32)
      acd32(4)=spvae1l4(iv2)
      acd32(5)=abb32(11)
      acd32(6)=spvae1l5(iv2)
      acd32(7)=abb32(21)
      acd32(8)=spvae1e2(iv2)
      acd32(9)=abb32(20)
      acd32(10)=spvak2e1(iv2)
      acd32(11)=spvae1l3(iv1)
      acd32(12)=spvae1l4(iv1)
      acd32(13)=spvae1l5(iv1)
      acd32(14)=spvae1e2(iv1)
      acd32(15)=spval4e1(iv2)
      acd32(16)=abb32(25)
      acd32(17)=spvae2e1(iv2)
      acd32(18)=abb32(24)
      acd32(19)=spval4e1(iv1)
      acd32(20)=spvae2e1(iv1)
      acd32(21)=spval3e1(iv2)
      acd32(22)=abb32(46)
      acd32(23)=spval3e1(iv1)
      acd32(24)=abb32(29)
      acd32(25)=abb32(26)
      acd32(26)=abb32(48)
      acd32(27)=-acd32(9)*acd32(14)
      acd32(28)=-acd32(5)*acd32(12)
      acd32(29)=acd32(13)*acd32(7)
      acd32(30)=-acd32(11)*acd32(3)
      acd32(27)=acd32(30)+acd32(29)+acd32(27)+acd32(28)
      acd32(27)=acd32(10)*acd32(27)
      acd32(28)=-acd32(8)*acd32(9)
      acd32(29)=-acd32(4)*acd32(5)
      acd32(30)=acd32(6)*acd32(7)
      acd32(31)=-acd32(2)*acd32(3)
      acd32(28)=acd32(31)+acd32(30)+acd32(28)+acd32(29)
      acd32(28)=acd32(1)*acd32(28)
      acd32(29)=-acd32(8)*acd32(26)
      acd32(30)=-acd32(4)*acd32(22)
      acd32(29)=acd32(30)+acd32(29)
      acd32(29)=acd32(23)*acd32(29)
      acd32(30)=-acd32(14)*acd32(26)
      acd32(31)=-acd32(12)*acd32(22)
      acd32(30)=acd32(30)+acd32(31)
      acd32(30)=acd32(21)*acd32(30)
      acd32(31)=-acd32(17)*acd32(25)
      acd32(32)=acd32(15)*acd32(24)
      acd32(31)=acd32(31)+acd32(32)
      acd32(31)=acd32(13)*acd32(31)
      acd32(32)=-acd32(17)*acd32(18)
      acd32(33)=acd32(15)*acd32(16)
      acd32(32)=acd32(32)+acd32(33)
      acd32(32)=acd32(11)*acd32(32)
      acd32(33)=-acd32(20)*acd32(25)
      acd32(34)=acd32(19)*acd32(24)
      acd32(33)=acd32(33)+acd32(34)
      acd32(33)=acd32(6)*acd32(33)
      acd32(34)=-acd32(18)*acd32(20)
      acd32(35)=acd32(16)*acd32(19)
      acd32(34)=acd32(34)+acd32(35)
      acd32(34)=acd32(2)*acd32(34)
      brack=acd32(27)+acd32(28)+acd32(29)+acd32(30)+acd32(31)+acd32(32)+acd32(3&
      &3)+acd32(34)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd32h8_qp
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
end module     p2_gg_httbar_d32h8l1d_qp
