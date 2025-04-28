module     p0_gg_gh_d9h1l1d_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity1d9h1l1d_qp.f90
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
   integer, private :: iv4
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(11) :: acd9
      complex(ki) :: brack
      acd9(1)=dotproduct(qshift,spvak2k1)
      acd9(2)=dotproduct(qshift,spvak2k3)
      acd9(3)=abb9(6)
      acd9(4)=abb9(8)
      acd9(5)=abb9(7)
      acd9(6)=dotproduct(qshift,spvak2l4)
      acd9(7)=abb9(10)
      acd9(8)=abb9(11)
      acd9(9)=acd9(6)*acd9(7)
      acd9(10)=acd9(1)*acd9(4)
      acd9(11)=-acd9(1)*acd9(3)
      acd9(11)=acd9(5)+acd9(11)
      acd9(11)=acd9(2)*acd9(11)
      acd9(9)=acd9(11)+acd9(10)-acd9(8)+acd9(9)
      brack=acd9(9)*acd9(2)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(14) :: acd9
      complex(ki) :: brack
      acd9(1)=spvak2k1(iv1)
      acd9(2)=dotproduct(qshift,spvak2k3)
      acd9(3)=abb9(6)
      acd9(4)=abb9(8)
      acd9(5)=spvak2k3(iv1)
      acd9(6)=dotproduct(qshift,spvak2k1)
      acd9(7)=abb9(7)
      acd9(8)=dotproduct(qshift,spvak2l4)
      acd9(9)=abb9(10)
      acd9(10)=abb9(11)
      acd9(11)=spvak2l4(iv1)
      acd9(12)=acd9(2)*acd9(3)
      acd9(12)=acd9(12)-acd9(4)
      acd9(12)=acd9(1)*acd9(12)
      acd9(13)=-acd9(9)*acd9(11)
      acd9(14)=acd9(3)*acd9(6)
      acd9(14)=-acd9(7)+acd9(14)
      acd9(14)=acd9(5)*acd9(14)
      acd9(12)=2.0_ki*acd9(14)+acd9(13)+acd9(12)
      acd9(12)=acd9(2)*acd9(12)
      acd9(13)=-acd9(9)*acd9(8)
      acd9(14)=-acd9(4)*acd9(6)
      acd9(13)=acd9(14)+acd9(10)+acd9(13)
      acd9(13)=acd9(5)*acd9(13)
      brack=acd9(12)+acd9(13)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(16) :: acd9
      complex(ki) :: brack
      acd9(1)=spvak2k1(iv1)
      acd9(2)=spvak2k3(iv2)
      acd9(3)=dotproduct(qshift,spvak2k3)
      acd9(4)=abb9(6)
      acd9(5)=abb9(8)
      acd9(6)=spvak2k1(iv2)
      acd9(7)=spvak2k3(iv1)
      acd9(8)=dotproduct(qshift,spvak2k1)
      acd9(9)=abb9(7)
      acd9(10)=spvak2l4(iv2)
      acd9(11)=abb9(10)
      acd9(12)=spvak2l4(iv1)
      acd9(13)=acd9(4)*acd9(3)
      acd9(13)=-acd9(5)+2.0_ki*acd9(13)
      acd9(14)=-acd9(1)*acd9(13)
      acd9(15)=acd9(11)*acd9(12)
      acd9(16)=-acd9(4)*acd9(8)
      acd9(16)=acd9(9)+acd9(16)
      acd9(16)=acd9(7)*acd9(16)
      acd9(14)=2.0_ki*acd9(16)+acd9(15)+acd9(14)
      acd9(14)=acd9(2)*acd9(14)
      acd9(13)=-acd9(6)*acd9(13)
      acd9(15)=acd9(11)*acd9(10)
      acd9(13)=acd9(15)+acd9(13)
      acd9(13)=acd9(7)*acd9(13)
      brack=acd9(13)+acd9(14)
   end function brack_3
!---#] function brack_3:
!---#[ function brack_4:
   pure function brack_4(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(9) :: acd9
      complex(ki) :: brack
      acd9(1)=spvak2k1(iv1)
      acd9(2)=spvak2k3(iv2)
      acd9(3)=spvak2k3(iv3)
      acd9(4)=abb9(6)
      acd9(5)=spvak2k1(iv2)
      acd9(6)=spvak2k3(iv1)
      acd9(7)=spvak2k1(iv3)
      acd9(8)=acd9(5)*acd9(3)
      acd9(9)=acd9(7)*acd9(2)
      acd9(8)=acd9(9)+acd9(8)
      acd9(8)=acd9(8)*acd9(6)
      acd9(9)=acd9(1)*acd9(2)*acd9(3)
      acd9(8)=acd9(9)+acd9(8)
      brack=2.0_ki*acd9(8)*acd9(4)
   end function brack_4
!---#] function brack_4:
!---#[ function brack_5:
   pure function brack_5(Q, mu2) result(brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd9h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(1) :: acd9
      complex(ki) :: brack
      brack=0.0_ki
   end function brack_5
!---#] function brack_5:
!---#[ function derivative:
   function derivative(mu2,i1,i2,i3,i4) result(numerator)
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd9h1_qp
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
      qshift = k2
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
end module     p0_gg_gh_d9h1l1d_qp
