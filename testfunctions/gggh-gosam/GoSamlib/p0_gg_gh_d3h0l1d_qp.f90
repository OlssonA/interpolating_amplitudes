module     p0_gg_gh_d3h0l1d_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity0d3h0l1d_qp.f90
   ! generator: buildfortran_d.py
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_util_qp, only: cond, d => metric_tensor
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
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd3h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd3
      complex(ki) :: brack
      acd3(1)=abb3(9)
      brack=acd3(1)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd3h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(21) :: acd3
      complex(ki) :: brack
      acd3(1)=k2(iv1)
      acd3(2)=abb3(22)
      acd3(3)=k3(iv1)
      acd3(4)=abb3(16)
      acd3(5)=spvak1k3(iv1)
      acd3(6)=abb3(15)
      acd3(7)=spvak2k1(iv1)
      acd3(8)=abb3(12)
      acd3(9)=spvak2k3(iv1)
      acd3(10)=abb3(11)
      acd3(11)=spvak2l4(iv1)
      acd3(12)=abb3(13)
      acd3(13)=spval4k3(iv1)
      acd3(14)=abb3(19)
      acd3(15)=acd3(2)*acd3(1)
      acd3(16)=acd3(4)*acd3(3)
      acd3(17)=acd3(6)*acd3(5)
      acd3(18)=acd3(8)*acd3(7)
      acd3(19)=acd3(10)*acd3(9)
      acd3(20)=-acd3(12)*acd3(11)
      acd3(21)=acd3(14)*acd3(13)
      brack=acd3(15)+acd3(16)+acd3(17)+acd3(18)+acd3(19)+acd3(20)+acd3(21)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd3h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(20) :: acd3
      complex(ki) :: brack
      acd3(1)=d(iv1,iv2)
      acd3(2)=abb3(20)
      acd3(3)=k3(iv1)
      acd3(4)=spvak2k3(iv2)
      acd3(5)=abb3(18)
      acd3(6)=k3(iv2)
      acd3(7)=spvak2k3(iv1)
      acd3(8)=abb3(10)
      acd3(9)=spvak1k2(iv2)
      acd3(10)=abb3(21)
      acd3(11)=spvak1k2(iv1)
      acd3(12)=spvak1k3(iv1)
      acd3(13)=spvak2k1(iv2)
      acd3(14)=abb3(17)
      acd3(15)=spvak1k3(iv2)
      acd3(16)=spvak2k1(iv1)
      acd3(17)=-acd3(6)*acd3(5)
      acd3(18)=acd3(8)*acd3(4)
      acd3(19)=acd3(9)*acd3(10)
      acd3(17)=acd3(19)+2.0_ki*acd3(18)+acd3(17)
      acd3(17)=acd3(7)*acd3(17)
      acd3(18)=-acd3(3)*acd3(5)
      acd3(19)=acd3(11)*acd3(10)
      acd3(18)=acd3(19)+acd3(18)
      acd3(18)=acd3(4)*acd3(18)
      acd3(19)=acd3(13)*acd3(12)
      acd3(20)=acd3(16)*acd3(15)
      acd3(19)=acd3(20)+acd3(19)
      acd3(19)=acd3(14)*acd3(19)
      acd3(20)=acd3(2)*acd3(1)
      brack=acd3(17)+acd3(18)+acd3(19)+2.0_ki*acd3(20)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd3h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd3
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_4
!---#] function brack_4:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3) result(numerator)
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd3h0_qp
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
end module     p0_gg_gh_d3h0l1d_qp
