module     p2_gg_httbar_d5h0l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d5h0l1d_qp.f90
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
      use p2_gg_httbar_abbrevd5h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(40) :: acd5
      complex(ki) :: brack
      acd5(1)=dotproduct(qshift,qshift)
      acd5(2)=abb5(15)
      acd5(3)=dotproduct(qshift,spval4k2)
      acd5(4)=abb5(12)
      acd5(5)=dotproduct(qshift,spval4l3)
      acd5(6)=abb5(6)
      acd5(7)=dotproduct(qshift,spval5k2)
      acd5(8)=abb5(20)
      acd5(9)=dotproduct(qshift,spval5l3)
      acd5(10)=abb5(7)
      acd5(11)=dotproduct(qshift,spvak2e1)
      acd5(12)=abb5(10)
      acd5(13)=dotproduct(qshift,spvak2e2)
      acd5(14)=abb5(14)
      acd5(15)=dotproduct(qshift,spval3e1)
      acd5(16)=abb5(18)
      acd5(17)=dotproduct(qshift,spvae1l3)
      acd5(18)=abb5(11)
      acd5(19)=dotproduct(qshift,spval3e2)
      acd5(20)=abb5(17)
      acd5(21)=dotproduct(qshift,spvae2l3)
      acd5(22)=abb5(16)
      acd5(23)=dotproduct(qshift,spvae1e2)
      acd5(24)=abb5(9)
      acd5(25)=dotproduct(qshift,spvae2e1)
      acd5(26)=abb5(13)
      acd5(27)=abb5(8)
      acd5(28)=acd5(2)*acd5(1)
      acd5(29)=-acd5(4)*acd5(3)
      acd5(30)=-acd5(6)*acd5(5)
      acd5(31)=-acd5(8)*acd5(7)
      acd5(32)=-acd5(10)*acd5(9)
      acd5(33)=-acd5(12)*acd5(11)
      acd5(34)=-acd5(14)*acd5(13)
      acd5(35)=-acd5(16)*acd5(15)
      acd5(36)=-acd5(18)*acd5(17)
      acd5(37)=-acd5(20)*acd5(19)
      acd5(38)=-acd5(22)*acd5(21)
      acd5(39)=-acd5(24)*acd5(23)
      acd5(40)=-acd5(26)*acd5(25)
      brack=acd5(27)+acd5(28)+acd5(29)+acd5(30)+acd5(31)+acd5(32)+acd5(33)+acd5&
      &(34)+acd5(35)+acd5(36)+acd5(37)+acd5(38)+acd5(39)+acd5(40)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd5h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(39) :: acd5
      complex(ki) :: brack
      acd5(1)=qshift(iv1)
      acd5(2)=abb5(15)
      acd5(3)=spval4k2(iv1)
      acd5(4)=abb5(12)
      acd5(5)=spval4l3(iv1)
      acd5(6)=abb5(6)
      acd5(7)=spval5k2(iv1)
      acd5(8)=abb5(20)
      acd5(9)=spval5l3(iv1)
      acd5(10)=abb5(7)
      acd5(11)=spvak2e1(iv1)
      acd5(12)=abb5(10)
      acd5(13)=spvak2e2(iv1)
      acd5(14)=abb5(14)
      acd5(15)=spval3e1(iv1)
      acd5(16)=abb5(18)
      acd5(17)=spvae1l3(iv1)
      acd5(18)=abb5(11)
      acd5(19)=spval3e2(iv1)
      acd5(20)=abb5(17)
      acd5(21)=spvae2l3(iv1)
      acd5(22)=abb5(16)
      acd5(23)=spvae1e2(iv1)
      acd5(24)=abb5(9)
      acd5(25)=spvae2e1(iv1)
      acd5(26)=abb5(13)
      acd5(27)=acd5(2)*acd5(1)
      acd5(28)=acd5(4)*acd5(3)
      acd5(29)=acd5(6)*acd5(5)
      acd5(30)=acd5(8)*acd5(7)
      acd5(31)=acd5(10)*acd5(9)
      acd5(32)=acd5(12)*acd5(11)
      acd5(33)=acd5(14)*acd5(13)
      acd5(34)=acd5(16)*acd5(15)
      acd5(35)=acd5(18)*acd5(17)
      acd5(36)=acd5(20)*acd5(19)
      acd5(37)=acd5(22)*acd5(21)
      acd5(38)=acd5(24)*acd5(23)
      acd5(39)=acd5(26)*acd5(25)
      brack=-2.0_ki*acd5(27)+acd5(28)+acd5(29)+acd5(30)+acd5(31)+acd5(32)+acd5(&
      &33)+acd5(34)+acd5(35)+acd5(36)+acd5(37)+acd5(38)+acd5(39)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd5h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(3) :: acd5
      complex(ki) :: brack
      acd5(1)=d(iv1,iv2)
      acd5(2)=abb5(15)
      brack=2.0_ki*acd5(2)*acd5(1)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd5h0_qp
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
end module     p2_gg_httbar_d5h0l1d_qp
